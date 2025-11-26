@echo off
for /r %%i in (Main_Seminar_LLM_LangChain_*.tex) do texify -cp %%i
