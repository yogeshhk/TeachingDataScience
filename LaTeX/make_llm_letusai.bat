@echo off
for /r %%i in (Main_Seminar_LLM_Transformers*.tex) do texify -cp %%i
