@echo off
for /r %%i in (*_ChatGPT*.tex) do texify -cp %%i
