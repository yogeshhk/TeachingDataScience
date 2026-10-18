@echo off
for /r %%i in (Main_Seminar_LLM_LangGraph_Presentation*.tex) do texify -cp %%i
