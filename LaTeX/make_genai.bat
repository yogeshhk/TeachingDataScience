@echo off
for /r %%i in (Main_Seminar_GenAI*.tex) do texify -cp %%i
