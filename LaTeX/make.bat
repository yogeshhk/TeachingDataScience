@echo off
for /r %%i in (Main_Workshop_*LangGraph*.tex) do texify -cp %%i
