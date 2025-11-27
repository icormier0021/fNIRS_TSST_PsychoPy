#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.1.4),
    on Thu Nov 27 09:52:40 2025
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

import psychopy
psychopy.useVersion('2024.1.4')


# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard

# Run 'Before Experiment' code from end_tsst_instruction
from psychopy.hardware import keyboard
# for markers
import socket
from pylsl import StreamInfo, StreamOutlet
markers = StreamInfo('fNIRSMarkerStream', 'Markers', 1, 0, 'int32', 'CPSPsychoPy')
lsl_outlet = StreamOutlet(markers)
modality = 'nirs'
# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.1.4'
expName = 'TSST_fnirs_V4_ARITH'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1440, 900]
_loggingLevel = logging.getLevel('info')
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # override logging level
    _loggingLevel = logging.getLevel(
        prefs.piloting['pilotLoggingLevel']
    )

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='/Users/Isaac/Documents/fNIRS_TSST_PsychoPy/builder_files/TSST_fnirs_V4_ARITH.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # this outputs to the screen, not a file
    logging.console.setLevel(_loggingLevel)
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log', level=_loggingLevel)
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowStencil=False,
            monitor='testMonitor', color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height', 
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [-1.0000, -1.0000, -1.0000]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.mouseVisible = False
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    ioSession = ioServer = eyetracker = None
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ptb'
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # prevent components from auto-drawing
    win.stashAutoDraw()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='PsychToolbox',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # flip the screen
        win.flip()
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # restore auto-drawn components
    win.retrieveAutoDraw()
    # reset any timers
    for timer in timers:
        timer.reset()


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='PsychToolbox'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "TSST_instruction_start" ---
    tsst_instruction_text = visual.TextStim(win=win, name='tsst_instruction_text',
        text='The judges will explain the task to you.\n\nPlease do not press any keys. ',
        font='Arial',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    # Run 'Begin Experiment' code from end_tsst_instruction
    kb = keyboard.Keyboard()
    # Run 'Begin Experiment' code from createTones
    import psychtoolbox as ptb
    from psychopy import sound
    
    arithTone = sound.Sound(800, secs=0.2, hamming=True)
    arithTone.setVolume(1.0)
    
    pauseTone = sound.Sound(600, secs=0.2, hamming=True)
    pauseTone.setVolume(1.0)
    
    # --- Initialize components for Routine "TSST_ARITH_TASK" ---
    subtract_number_TSST = visual.TextStim(win=win, name='subtract_number_TSST',
        text='',
        font='Arial',
        pos=(0, 0), height=0.07, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "TSST_ARITH_PAUSE" ---
    subtract_pausing_TSST = visual.TextStim(win=win, name='subtract_pausing_TSST',
        text='',
        font='Arial',
        pos=(0, 0), height=0.07, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "Finished" ---
    finished_text = visual.TextStim(win=win, name='finished_text',
        text='This part of the task is finished, please wait for further instructions.',
        font='Arial',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "TSST_instruction_start" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('TSST_instruction_start.started', globalClock.getTime(format='float'))
    # Run 'Begin Routine' code from end_tsst_instruction
    kb.clearEvents()
    win.mouseVisible = False
    # keep track of which components have finished
    TSST_instruction_startComponents = [tsst_instruction_text]
    for thisComponent in TSST_instruction_startComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "TSST_instruction_start" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *tsst_instruction_text* updates
        
        # if tsst_instruction_text is starting this frame...
        if tsst_instruction_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            tsst_instruction_text.frameNStart = frameN  # exact frame index
            tsst_instruction_text.tStart = t  # local t and not account for scr refresh
            tsst_instruction_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(tsst_instruction_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'tsst_instruction_text.started')
            # update status
            tsst_instruction_text.status = STARTED
            tsst_instruction_text.setAutoDraw(True)
        
        # if tsst_instruction_text is active this frame...
        if tsst_instruction_text.status == STARTED:
            # update params
            pass
        # Run 'Each Frame' code from end_tsst_instruction
        ## EACH FRAME
        
        continue_key = kb.getKeys(["return"], waitRelease=False, clear=True)
        # Check if space key was pressed
        if continue_key:
            continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in TSST_instruction_startComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "TSST_instruction_start" ---
    for thisComponent in TSST_instruction_startComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('TSST_instruction_start.stopped', globalClock.getTime(format='float'))
    # Run 'End Routine' code from end_tsst_instruction
    kb.clearEvents()
    thisExp.nextEntry()
    # the Routine "TSST_instruction_start" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    TSST_Arith_Loop = data.TrialHandler(nReps=1.0, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions("stims/TSST_arith.xlsx"),
        seed=None, name='TSST_Arith_Loop')
    thisExp.addLoop(TSST_Arith_Loop)  # add the loop to the experiment
    thisTSST_Arith_Loop = TSST_Arith_Loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTSST_Arith_Loop.rgb)
    if thisTSST_Arith_Loop != None:
        for paramName in thisTSST_Arith_Loop:
            globals()[paramName] = thisTSST_Arith_Loop[paramName]
    
    for thisTSST_Arith_Loop in TSST_Arith_Loop:
        currentLoop = TSST_Arith_Loop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisTSST_Arith_Loop.rgb)
        if thisTSST_Arith_Loop != None:
            for paramName in thisTSST_Arith_Loop:
                globals()[paramName] = thisTSST_Arith_Loop[paramName]
        
        # --- Prepare to start Routine "TSST_ARITH_TASK" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('TSST_ARITH_TASK.started', globalClock.getTime(format='float'))
        subtract_number_TSST.setText(StartingNumber)
        # Run 'Begin Routine' code from skip_TSST
        lsl_outlet.push_sample([50]) #TSST_ARITH_TASK Block Start
        skipNextRoutine = False
        win.mouseVisible = False
        #start_tone_3.status = NOT_STARTED
        
        #Schedule start tone to begin on next screen flip
        nextFlip_ptb = win.getFutureFlipTime(clock='ptb')
        arithTone.play(when=nextFlip_ptb)
        # keep track of which components have finished
        TSST_ARITH_TASKComponents = [subtract_number_TSST]
        for thisComponent in TSST_ARITH_TASKComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "TSST_ARITH_TASK" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 40.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *subtract_number_TSST* updates
            
            # if subtract_number_TSST is starting this frame...
            if subtract_number_TSST.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                subtract_number_TSST.frameNStart = frameN  # exact frame index
                subtract_number_TSST.tStart = t  # local t and not account for scr refresh
                subtract_number_TSST.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(subtract_number_TSST, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'subtract_number_TSST.started')
                # update status
                subtract_number_TSST.status = STARTED
                subtract_number_TSST.setAutoDraw(True)
            
            # if subtract_number_TSST is active this frame...
            if subtract_number_TSST.status == STARTED:
                # update params
                pass
            
            # if subtract_number_TSST is stopping this frame...
            if subtract_number_TSST.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > subtract_number_TSST.tStartRefresh + 40-frameTolerance:
                    # keep track of stop time/frame for later
                    subtract_number_TSST.tStop = t  # not accounting for scr refresh
                    subtract_number_TSST.tStopRefresh = tThisFlipGlobal  # on global time
                    subtract_number_TSST.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'subtract_number_TSST.stopped')
                    # update status
                    subtract_number_TSST.status = FINISHED
                    subtract_number_TSST.setAutoDraw(False)
            # Run 'Each Frame' code from skip_TSST
            skip_keys = kb.getKeys(['s'], waitRelease=False, clear=True)
            if skip_keys:
                TSST_Arith_Loop.finished = True
                skipNextRoutine = True 
                continueRoutine = False
            # Run 'Each Frame' code from check_time_2
            # hard stop at 40 s
            if t >= 40.0 - frameTolerance:
                subtract_number_TSST.setAutoDraw(False)
                continueRoutine = False
            
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in TSST_ARITH_TASKComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "TSST_ARITH_TASK" ---
        for thisComponent in TSST_ARITH_TASKComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('TSST_ARITH_TASK.stopped', globalClock.getTime(format='float'))
        # Run 'End Routine' code from skip_TSST
        lsl_outlet.push_sample([51]) #TSST_ARITH_TASK Block End
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-40.000000)
        
        # --- Prepare to start Routine "TSST_ARITH_PAUSE" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('TSST_ARITH_PAUSE.started', globalClock.getTime(format='float'))
        subtract_pausing_TSST.setText('PAUSE')
        # Run 'Begin Routine' code from check_time_4
        lsl_outlet.push_sample([60]) #TSST_ARITH_PAUSE Block Start
        win.mouseVisible = False
        
        if skipNextRoutine:
            skipNextRoutine = False
            continueRoutine = False
        else:
            #Schedule pause tone to begin on next screen flip
            nextFlip_ptb = win.getFutureFlipTime(clock='ptb')
            pauseTone.play(when=nextFlip_ptb)
        #tsst_start_pause_tone.status = NOT_STARTED
        # keep track of which components have finished
        TSST_ARITH_PAUSEComponents = [subtract_pausing_TSST]
        for thisComponent in TSST_ARITH_PAUSEComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "TSST_ARITH_PAUSE" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 20.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *subtract_pausing_TSST* updates
            
            # if subtract_pausing_TSST is starting this frame...
            if subtract_pausing_TSST.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                subtract_pausing_TSST.frameNStart = frameN  # exact frame index
                subtract_pausing_TSST.tStart = t  # local t and not account for scr refresh
                subtract_pausing_TSST.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(subtract_pausing_TSST, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'subtract_pausing_TSST.started')
                # update status
                subtract_pausing_TSST.status = STARTED
                subtract_pausing_TSST.setAutoDraw(True)
            
            # if subtract_pausing_TSST is active this frame...
            if subtract_pausing_TSST.status == STARTED:
                # update params
                pass
            
            # if subtract_pausing_TSST is stopping this frame...
            if subtract_pausing_TSST.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > subtract_pausing_TSST.tStartRefresh + 20-frameTolerance:
                    # keep track of stop time/frame for later
                    subtract_pausing_TSST.tStop = t  # not accounting for scr refresh
                    subtract_pausing_TSST.tStopRefresh = tThisFlipGlobal  # on global time
                    subtract_pausing_TSST.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'subtract_pausing_TSST.stopped')
                    # update status
                    subtract_pausing_TSST.status = FINISHED
                    subtract_pausing_TSST.setAutoDraw(False)
            # Run 'Each Frame' code from check_time_4
            # hard stop at 40 s
            if t >= 20.0 - frameTolerance:
                subtract_pausing_TSST.setAutoDraw(False)
                continueRoutine = False
            
            
            skip_keys = kb.getKeys(['s'],waitRelease=False, clear=True)
            if skip_keys:
                TSST_Arith_Loop.finished = True
                skipNextRoutine = True 
                continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in TSST_ARITH_PAUSEComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "TSST_ARITH_PAUSE" ---
        for thisComponent in TSST_ARITH_PAUSEComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('TSST_ARITH_PAUSE.stopped', globalClock.getTime(format='float'))
        # Run 'End Routine' code from check_time_4
        lsl_outlet.push_sample([61]) #TSST_ARITH_PAUSE Block End
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-20.000000)
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'TSST_Arith_Loop'
    
    
    # --- Prepare to start Routine "Finished" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('Finished.started', globalClock.getTime(format='float'))
    # Run 'Begin Routine' code from finished_code
    kb.clearEvents()
    win.mouseVisible = False
    # keep track of which components have finished
    FinishedComponents = [finished_text]
    for thisComponent in FinishedComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Finished" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *finished_text* updates
        
        # if finished_text is starting this frame...
        if finished_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            finished_text.frameNStart = frameN  # exact frame index
            finished_text.tStart = t  # local t and not account for scr refresh
            finished_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(finished_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'finished_text.started')
            # update status
            finished_text.status = STARTED
            finished_text.setAutoDraw(True)
        
        # if finished_text is active this frame...
        if finished_text.status == STARTED:
            # update params
            pass
        # Run 'Each Frame' code from finished_code
        ## EACH FRAME
        
        task_key = kb.getKeys(["return"], waitRelease=False, clear=True)
        # Check if space key was pressed
        if task_key:
            continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in FinishedComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Finished" ---
    for thisComponent in FinishedComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('Finished.stopped', globalClock.getTime(format='float'))
    # Run 'End Routine' code from finished_code
    kb.clearEvents()
    thisExp.nextEntry()
    # the Routine "Finished" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # shut down eyetracker, if there is one
    if deviceManager.getDevice('eyetracker') is not None:
        deviceManager.removeDevice('eyetracker')
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    # shut down eyetracker, if there is one
    if deviceManager.getDevice('eyetracker') is not None:
        deviceManager.removeDevice('eyetracker')
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
