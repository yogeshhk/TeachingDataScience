@echo off
for /r %%i in (Main_Seminar_LLM_AutoAgents_*.tex) do texify -cp %%i
