@echo off
for /r %%i in (Main_Seminar_Google*.tex) do texify -cp %%i
