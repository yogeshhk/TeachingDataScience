@echo off
for /r %%i in (Main_Seminar_AI_ClaudeCode*.tex) do texify -cp %%i
