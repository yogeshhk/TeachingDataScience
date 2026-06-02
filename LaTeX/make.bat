@echo off
for /r %%i in (Main_Seminar_Tech_Career*.tex) do texify -cp %%i
