@echo off
for /r %%i in (Main_Seminar_LLM_LangChain*.tex) do texify -cp %%i
