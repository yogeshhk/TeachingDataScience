@echo off
for /r %%i in (Main_Seminar_Tech_CareerInDataScience_Presentation.tex) do texify -cp %%i
