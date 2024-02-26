@echo off
for /r %%i in (Main_Seminar_LLM_Lang*.tex) do texify -cp %%i
