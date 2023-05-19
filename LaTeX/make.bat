@echo off
for /r %%i in (Main_*Prompt*.tex) do texify -cp %%i
