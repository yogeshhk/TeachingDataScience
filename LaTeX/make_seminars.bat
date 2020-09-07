@echo off
for /r %%i in (Main_Seminar*.tex) do texify -cp %%i
