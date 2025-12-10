@echo off
for /r %%i in (Main_Seminar_LLM_GenAI*.tex) do texify -cp %%i
