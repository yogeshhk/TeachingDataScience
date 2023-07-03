@echo off
for /r %%i in (Main*LLM*.tex) do texify -cp %%i
