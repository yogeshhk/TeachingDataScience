@echo off
for /r %%i in (Main_Seminar_LLM*.tex) do texify -cp %%i
