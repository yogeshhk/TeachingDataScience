@echo off
for /r %%i in (Main_Seminar_OpenCode*.tex) do texify -cp %%i
