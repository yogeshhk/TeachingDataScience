@echo off
for /r %%i in (Main_Workshop_LLM_Presentation*.tex) do texify -cp %%i
