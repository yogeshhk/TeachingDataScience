@echo off
for /r %%i in (Main_*Graph_KnowledgeGraph_*.tex) do texify -cp %%i
