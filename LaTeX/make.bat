@echo off
for /r %%i in (Main_Seminar_LLM_Agents_Presentation*.tex) do texify -cp %%i
