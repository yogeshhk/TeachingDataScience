@echo off
for /r %%i in (Main_Seminar_*Parsing*.tex) do texify -cp %%i
