# GUI Screenshots

Screenshots for the [MATLAB GUI Tutorial](../MATLAB_GUI_Tutorial.md) are saved here.

## Generating screenshots

From the project root in MATLAB:

```matlab
run_prismt_gui
% Perform each step, then:
addpath('scripts')
capture_gui_screenshots('1_initial')   % After launch
capture_gui_screenshots('2_path_entered')  % After Browse
capture_gui_screenshots('3_loaded')     % After Load
capture_gui_screenshots('4_conditions') % After selecting conditions
capture_gui_screenshots('5_ready')     % Ready to run
```

Screenshots are saved to `docs/images/gui/`.
