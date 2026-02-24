@echo off
for /r %%i in (Main_Seminar_AI_Tools*.tex) do texify -cp %%i
