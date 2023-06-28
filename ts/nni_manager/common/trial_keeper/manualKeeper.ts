// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/**
 *  The helper class to implement a "reuse" training service.
 *
 *  For reuse training services, each machine (or VM) will have a long run daemon, the trial keeper.
 *  Trial keeper takes responsibility for:
 *
 *   1. Spawning and killing trial processes.
 *   2. Communicating with trials to send/recieve parameters and metrics.
 *   3. (WIP) Upload and download trial files (trial code, logs, user defined output files, etc).
 *
 *  The trial keeper will have a command channel to communicate with the training service.
 *  The channel protocol and direction (who is server) will be decided by the training service.
 *
 *  The design philosophy is to minimize the differences among reuse training services.
 *  Each training service only needs to launch the trial keeper and establish the command channel,
 *  and then all other works can be done with sending and receiving uniformed commands.
 *
 *  TrialKeeper has a very similar interface to TrainingSerivceV3.
 *  Check it for method parameters' definition.
 **/

import { EventEmitter } from 'events';
import fs from 'fs/promises';
import path from 'path';

import tar from 'tar';

import type { Command } from 'common/command_channel/interface';
import { HttpChannelServer } from 'common/command_channel/http';
import globals from 'common/globals';
import { Logger, getLogger } from 'common/log';
import type { EnvironmentInfo } from 'common/training_service_v3';
import { collectPlatformInfo } from './collect_platform_info';
// import { TrialProcess, TrialProcessOptions } from './process';
import { TaskSchedulerClient } from './task_scheduler_client';
import { assert } from 'console';

export declare namespace ManualTrialKeeper {
    export interface TrialOptions {
        id: string;
        command: string;
        codeDirectoryName: string;
        sequenceId?: number;
        gpuNumber?: number;
        gpuRestrictions?: GpuRestrictions;
    }

    export interface GpuRestrictions {
        onlyUseIndices?: number[];
        rejectActive?: boolean;
        rejectComputeActive?: boolean;
    }
}

export class ManualTrialKeeper {
    private envId: string;
    private envInfo!: EnvironmentInfo;
    private channels: HttpChannelServer;
    private trial_info_channel: HttpChannelServer;
    private dirs: Map<string, string> = new Map();
    private emitter: EventEmitter = new EventEmitter();
    private scheduler: TaskSchedulerClient;
    private log: Logger;
    private platform: string;
    // private trials: Map<string, TrialProcess> = new Map();
    private trialIds: Map<string, string> = new Map();
    private gpuEnabled: boolean;

    constructor(environmentId: string, platform: string, enableGpuScheduling: boolean) {
        this.envId = environmentId;
        this.platform = platform;
        this.gpuEnabled = enableGpuScheduling;
        this.log = getLogger(`TrialKeeper.${environmentId}`);

        this.scheduler = new TaskSchedulerClient(enableGpuScheduling);
        this.scheduler.onUtilityUpdate(info => {
            Object.assign(this.envInfo, info);
            this.emitter.emit('env_update', this.envInfo);
        });

        this.channels = new HttpChannelServer(this.envId, `/env/${this.envId}`);
        this.channels.onReceive((trialId, command) => {
            this.emitter.emit('command', trialId, command);
            if (command.type !== 'request_parameter' && command.type !== 'metric') {
                this.log.warning(`Unexpected command from trial ${trialId}:`, command);
            }
        });

        // http communication channel for trial start and stop
        this.trial_info_channel = new HttpChannelServer(this.envId, `/trial-info/${this.envId}`);
        this.trial_info_channel.onReceive((channelId, command) => {
            this.log.trace(channelId)
            if (command.type === 'request_trial_info') {
                this.emitter.emit('request_trial_info', this.trialIds.get(channelId), Date.now());
                this.emitter.emit('trial_start', this.trialIds.get(channelId), Date.now())
            }
            if (command.type === 'request_trial_stop') {
                this.emitter.emit('request_trial_stop', command.trialId, Date.now(), command.exitCode);
                this.emitter.emit('trial_stop', command.trialId, Date.now(), command.exitCode);
                this.trial_info_channel.send('stop', {});
            }
            else{
                this.log.warning(`Unexpected command from http trial ${command.trialId}:`, command);
            }
        });
    }

    // TODO: support user configurable init command
    public async start(): Promise<EnvironmentInfo> {
        this.envInfo = { id: this.envId, type: 'hot' } as EnvironmentInfo;

        await Promise.all([
            this.scheduler.start(),
            this.channels.start(),
            this.trial_info_channel.start()
        ]);

        Object.assign(this.envInfo, await collectPlatformInfo(this.gpuEnabled));
        return this.envInfo;
    }

    public async shutdown(): Promise<void> {
        let promises: Promise<void>[] = [
            this.scheduler.shutdown(),
            this.channels.shutdown(),
            this.trial_info_channel.shutdown()
        ];

        // const trials = Array.from(this.trials.values());
        // promises = promises.concat(trials.map(trial => trial.kill()));

        await Promise.all(promises);
    }

    public registerDirectory(name: string, path: string): void {
        this.dirs.set(name, path);
    }

    public async unpackDirectory(name: string, tarPath: string): Promise<void> {
        const extractDir = path.join(
            globals.paths.experimentRoot, 
            'environments',
            (globals.args as any).environmentId,
            'upload',
            name
        );
        await fs.mkdir(extractDir, { recursive: true });
        await tar.extract({ cwd: extractDir, file: tarPath });

        this.registerDirectory(name, extractDir);
    }

    
    public async createTrial(options: ManualTrialKeeper.TrialOptions): Promise<boolean> {
        this.trialIds.set(String(options.sequenceId ?? -1), options.id)
        
        console.trace('HttpKeeper trialIds:', this.trialIds);
        const trialId = options.id;

        this.log.debug('Httpkeeper createTrial start')

        const gpuEnv = await this.scheduler.schedule(trialId, options.gpuNumber, options.gpuRestrictions);
        if (gpuEnv === null) {
            // TODO: should scheduler report concrete fail reason?
            this.log.info('Scheduling failed because the GPU constraint cannot be satisfied');
            return false;
        }

        // TODO: move this to globals.paths
        const outputDir = path.join(globals.paths.experimentRoot, 'environments', this.envId, 'trials', trialId);
        await fs.mkdir(outputDir, { recursive: true });

        // const trial = new TrialProcess(trialId);
        // trial.onStart(timestamp => {
        //     this.emitter.emit('trial_start', trialId, timestamp);
        // });
        // trial.onStop((timestamp, exitCode, _signal) => {
        //     this.emitter.emit('trial_stop', trialId, timestamp, exitCode);
        //     this.scheduler.release(trialId);  // TODO: fire and forget, handle exception?
        // });

        const env: Record<string, string> = {...gpuEnv };
        env['NNI_CODE_DIR'] = this.dirs.get(options.codeDirectoryName)!;
        env['NNI_EXP_ID'] = globals.args.experimentId;
        env['NNI_OUTPUT_DIR'] = outputDir;
        env['NNI_PLATFORM'] = this.platform;
        env['NNI_SYS_DIR'] = outputDir;
        env['NNI_TRIAL_COMMAND_CHANNEL'] = this.channels.getChannelUrl(trialId);
        env['NNI_TRIAL_HTTP_INFO_CHANNEL'] = this.trial_info_channel.getChannelUrl(String(options.sequenceId ?? -1));
        env['NNI_TRIAL_JOB_ID'] = trialId;
        env['NNI_TRIAL_SEQ_ID'] = String(options.sequenceId ?? -1);
        
        const command = {type: 'trial_info', env};

        this.trial_info_channel.send(String(options.sequenceId ?? -1), command);
        this.trial_info_channel.send('stop', {});
        this.emitter.on('request_trial_info', this._trial_recive_requset_info);
        this.emitter.on('request_trial_stop', this._trial_recive_requset_stop);
        console.trace('trial_info_channel command',command);

        // const procOptions: TrialProcessOptions = {
        //     command: options.command,
        //     codeDirectory: this.dirs.get(options.codeDirectoryName)!,
        //     outputDirectory: outputDir,
        //     commandChannelUrl: this.channels.getChannelUrl(trialId),
        //     platform: this.platform,
        //     sequenceId: options.sequenceId,
        //     environmentVariables: gpuEnv,
        // }

        // const success = await trial.spawn(procOptions);

        // console.trace('createTrial trial_info_channel',this.trial_info_channel.getChannelUrl(''))
        // console.trace('createTrial procOptions',procOptions)

        const success = true;
        if (success) {
            // this.trials.set(trialId, trial);
            return true;
        } else {
            return false;
        }
    }
    public async stopTrial(trialId: string): Promise<void> {
        // await this.trials.get(trialId)!.kill();
        await this.trialIds.get(trialId);
    }

    public async sendCommand(trialId: string, command: Command): Promise<void> {
        this.channels.send(trialId, command);
    }

    public _trial_recive_requset_info(trialId: string, timestamp: number): void{
        console.trace('HttpKeeper recived requset for trial info and start:', trialId, timestamp)
    }
    public _trial_recive_requset_stop(trialId: string, timestamp: number, exitCode: number | null): void{
        console.trace('HttpKeeper recived stop:', trialId, timestamp, exitCode)
    }

    public onTrialStart(callback: (trialId: string, timestamp: number) => void): void {
        this.emitter.on('trial_start', callback);
    }

    public onTrialStop(callback: (trialId: string, timestamp: number, exitCode: number | null) => void): void {
        this.emitter.on('trial_stop', callback);
    }

    public onReceiveCommand(callback: (trialId: string, command: Command) => void): void;
    public onReceiveCommand(commandType: string, callback: (trialId: string, command: Command) => void): void;

    public onReceiveCommand(commandTypeOrCallback: any, callbackOrNone?: any): void {
        if (callbackOrNone) {
            this.emitter.on('command', (trialId, command) => {
                if (command.type === commandTypeOrCallback) {
                    callbackOrNone(trialId, command);
                }
            });
        } else {
            this.emitter.on('command', commandTypeOrCallback);
        }
    }

    public onEnvironmentUpdate(callback: (info: EnvironmentInfo) => void): void {
        this.emitter.on('env_update', callback);
    }
}
