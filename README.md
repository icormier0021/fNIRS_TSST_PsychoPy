# Trier Social Stress Test (TSST) Experiment for fNIRS Study
**Prepared By**: Isaac Cormier
**Date**: 2025/08/06

# Overview
This is a pilot build of the Trier Social Stress Test (TSST) prepared in PsychoPy version 2024.1.4, based on the original TSST developed by Kirschbaum et al. (1993) and the methods published by Rosenbaum et al. (2018). This paradigm was created for the upcoming joint-fNIRS study between Dr. Jennifer Khoury's Developmental Psychobiology Research Centre (DPRC) at Mount Saint Vincent University and Dr. Aaron Newman's NeuroCognitive Imaging Lab at Dalhousie University. 

# Installation
Make sure you have PsychoPy version 2024.2.1. If using a newer version of PsychoPy, please make sure to change the version under 'Preferences'. This experiment was built and tested on a standalone installation, although users might have some success with other installation methods (for more details, visit: https://www.psychopy.org/download.html).


# Usage
There are three experiment files you can choose from:
1. 'TSST_fnirs_V2_FULL.psyexp': Contains all two control conditions and the TSST arithmetic condition. 
2. 'TSST_fnirs_V2_CONTROL.psyexp': Contains only the two control conditions. 
3. 'TSST_fnirs_V2_ARITH.psyexp': Contains only the TSST arithmetic condition.

Make sure that the experiment is set to 'Run' before starting the paradigm to ensure it is fullscreen. 

Before the experiment begins, you will need to enter a participant ID and session number. These will be auto-populated with default values, and can be modified as needed.

There are three primary routines: CTL1, CTL2, and the TSST_arith. All three routines present tones at the start and end of each task duration. The start and end tones are slightly different to allow for differentiation. If you do not hear any sound, you may need to go into the settings for each individual sound component and manually select the desired output speaker. 

To exit the paradigm, the user can press the 'esc' key at any time. To skip to the next routine, the user can press the 's' key. To continue from any instruction screen, press the ENTER or RETURN key.

The presented numbers and subtractions can be found in the document "TSST_answer_sheet.docx", which is used by the research assistants to follow along during the tasks and redirect the subject as needed when incorrect answers are provided.
 
Below is a brief overview of the paradigm, with descriptions and notes for each routine:

1. CTL1_instruction_start
- Presents the following text and waits for the user to press the 'ENTER' key to continue onto the next routine once instructed by the researchers:
"Please read the number that appears on the screen out loud. You can read at your own pace. Press the space bar to go to the next number. You will read out loud for 40 seconds and then a PAUSE screen will appear for 20 seconds, please take a break during the PAUSE. Press the ENTER KEY to start the task"

2. CTL1_Loop & CTL1
- CTL1_Loop is set to repeat the CTL1 routine for 6 blocks. 
- Each block of CTL1 includes 40 seconds of reading numbers (task), followed by 20 seconds of looking at a fixation cross (pause).
- The subject reads each number as it is presented on the screen. When finished repeating the number out loud, they must press the 'space' key to continue onto the next number. 
-The starting number is 1022, with the number reducing in steps of 13 after each 'space' key press. The last presented number before the fixation cross is the same number presented at the start of the subsequent block.
- Once the participant reaches the last number they can subtract from before reaching zero, the current number is set back to the starting number. 
- The use of the 'space' key allows the user to proceed at their own pace without time or social pressure.

3. CTL2_instruction_start
- Presents the following text and waits for the user to press the 'ENTER' key to continue onto the next routine once instructed by the researchers:
"Out loud, please consecutively subtract the number 13 from the number that appears on the screen. Keep subtracting the number 13 from the number.  If you reach 0, start again. If you make a mistake, the experimenter will tell you the correct number. You can work as fast or a slow as you want, just do your best! Please pause when you see the PAUSE screen. Press the ENTER KEY to start the task."

4. CTL2_Loop & CTL2
- CTL2_Loop pulls the the variables 'Block' and 'StartingNumber' from the excel sheet "CTL2.xlsx" found under the "stims" sub-directory. The loop is set to repeat 6 times, with each row of the excel sheet representing a different block. This allows the experimenter to change the starting numbers for each block as desired. 
- Each block of CTL2 includes 40 seconds of subtracting numbers out loud (task), followed by 20 seconds of looking at a fixation cross (pause).
- The subject is presented with the starting number, which remains on the screen for the duration of the task. They are instructed to perform serial subtractions without any aids in steps of 13. Each block has a different starting number (randomly selected by the paradigm developer).
- The subject should be instructed to take as much time as they like for the calculation. When errors are made, a research assistant should say "Okay, thank you. Please count now from...", giving them the right number without indicating errors. 

5. task_end
- Presents the text "Please wait for the next task to begin.." to signify the end of the control tasks and waits for the user to press the 'ENTER' key to continue onto the TSST once instructed by the researchers.
- After the completion of CTL1 and CTL2, research assistants designated as the TSST committee members dressed in white physician coats (one male and one female) will introduce themselves to the subject. The subjects will then undergo an anticipatory stress phase and mock job interview phase before starting the TSST arithmetic task.

6. TSST_instruction_start
- Presents the following text and waits for the judges to explain the instructions. This time, one of the research assistants, not the user, will press the 'ENTER' key to continue onto the next routine:
"The judges will explain the task to you. Please do not press any keys."

7. TSST_Arith_Loop & TSST_arith
- TSST_Arith_Loop pulls the the variables 'Block' and 'StartingNumber' from the excel sheet "TSST_arith.xlsx" found under the "stims" sub-directory. The loop is set to repeat 6 times, with each row of the excel sheet representing a different block.  This allows the experimenter to change the starting numbers for each block as desired. All starting numbers differ from those in CTL2.
- Each block of TSST_arith includes 40 seconds of subtracting numbers out loud (task), followed by 20 seconds of looking at a fixation cross (pause).
- The subject is presented with the starting number, which remains on the screen for the duration of the task. They are instructed to perform serial subtractions without any aids in steps of 13. Each block has a different starting number (randomly selected by the paradigm developer). 
- The subject is told that this task is independent from all prior tasks. To maximize social stress, subjects should be told to count as fast and correctly as possible, as well as to hold eye contact with one of the committee members. If an error occurs, the other committee member should say "Stop. Start again from...". 
- Before Blocks 3 and 5, subjects should be told to work faster and better. 

8. Finished
- Presents the following text and waits for the user to press the 'ENTER' key to continue onto the next routine once instructed by the researchers:
"This part of the task is finished, please wait for further instructions."

# Notes
- While based heavily off of the methods published by Rosenbaum et al. (2018), changes were made to adapt the in-person physical experiment setup to the computer-based PsychoPy setup. 


# Change Log
Version 1: 2025/08/05
- Completed pilot build
- Missing LabStreamingLayer integration

Version 2: 2025/08/22
- Broke original build up into three experiment versions: a full build with all conditions (TSST_fnirs_V2_FULL.psyexp), a build with only the control conditions (TSST_fnirs_V2_CONTROL.psyexp), and a build with only the TSST Arith condition (TSST_fnirs_V2_ARITH.psyexp).
- Replaced fixation cross on the pause screen with the text "PAUSE"
- Changed CTRL1 to restart from the initial number when reaching zero instead of presenting negative numbers.
- Created separate instruction screens before each block. 
- Fixed issue with prolonged length of CTl2 and TSST_arith conditions by changing number of repeats from '6' to '1'. Since the conditions spreadsheets contain 6 rows, number of repeats only needs to be '1' for all 6 rows to be shown (6 minutes total).
- Added code component to ensure clean transition between number and 'PAUSE' screen, as well as to ensure that each block is exactly 60 seconds.
- LabStreamingLayer to be incorporated once build is finalized.

Version 3: 2025/09/24
- Changed use of 'space' bar to 'ENTER' or 'RETURN' key when proceeding from instruction screens; this way, the space key is unique to the CTRL_1 task. 
- Changed all instruction text to be at a height of 0.5, consistent throughout the paradigm.
- Added a 'Finished' text screen at end of Arith TSST.
- Split up the CTRL1, CTRL2, and TSST Arith routines into two 'TASK' and 'PAUSE' routines for better programming of LabStreamingLayer.

Version 4: 2025/10/16
- Incorporated labstreaming layer into code for fNIRS and generated python scripts for easy use in Newman's lab.
- Moved builder scripts to separate 'builder_files' sub-directory.
- Created fNIRS_marker_key.json file with marker descriptions and response value assignments.
