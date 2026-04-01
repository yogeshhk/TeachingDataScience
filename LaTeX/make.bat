@echo off
for /r %%i in (Main_Seminar_*ClaudeCode*.tex) do texify -cp %%i
