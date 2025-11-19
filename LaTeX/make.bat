@echo off
for /r %%i in (Main_Seminar_Tech_CareerInDataScience_*.tex) do texify -cp %%i
