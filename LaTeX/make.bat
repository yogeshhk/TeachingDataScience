@echo off
for /r %%i in (Main_Seminar_Python_*.tex) do texify -cp %%i
