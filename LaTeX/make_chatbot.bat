@echo off
for /r %%i in (Main_*_Chatbot*.tex) do texify -cp %%i
