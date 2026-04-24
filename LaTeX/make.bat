@echo off
for /r %%i in (Main_Seminar_AI_ClaudeCode_Ch*.tex) do texify -cp %%i
