@echo off
for /r %%i in (Main_Workshop_Chatbot*.tex) do texify -cp %%i
