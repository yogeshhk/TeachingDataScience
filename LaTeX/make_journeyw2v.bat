@echo off
for /r %%i in (Main_Seminar_Journey*.tex) do texify -cp %%i
