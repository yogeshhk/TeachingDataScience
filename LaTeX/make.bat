@echo off
for /r %%i in (Main_Agenda_RAG_*.tex) do texify -cp %%i
