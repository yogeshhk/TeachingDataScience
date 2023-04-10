@echo off
for /r %%i in (Main*ChatGPT*.tex) do texify -cp %%i
