# 15 Puzzle solver

The search uses IDA* algorithm with walking distance heuristic - you have to run the following command at the start
```prolog
create_wd_database.
```
to generate all 24964 possible boards for this heuristic (otherwise it won't work) - it shouldn't take a bit more than a minute on a decent computer.

You can also use a much weaker heuristic (which doesn't have to generate anything) `MAX(inversion distance, manhattan)` but you have to change the `heuristic` in the code.

Then you can simply run 
```prolog
solve([1,10,15,4,0,13,6,8,2,9,3,7,14,5,12,11],  M).
```
(I recommend to do it like this though - so you can see the time and length of the solution)
```prolog
time((solve([1,10,15,4,0,13,6,8,2,9,3,7,14,5,12,11],  M), write(M), length(M,L))).
```

The solvabilty of the particular configuration is tested at the start of the `solve`.
You can also try to run prepared tests:
- `test0` - unsolvable
- `test1` ... `test6` - each shouldn't take more than a couple of minutes at max

### Interesting and helpful information about Walking distance and 15 puzzle in general:

- [https://michael.kim/blog/puzzle](https://michael.kim/blog/puzzle)
- [http://kociemba.org/themen/fifteen/fifteensolver.html](http://kociemba.org/themen/fifteen/fifteensolver.html)
- [https://web.archive.org/web/20141224035932/http://juropollo.xe0.ru/stp_wd_translation_en.htm](https://web.archive.org/web/20141224035932/http://juropollo.xe0.ru/stp_wd_translation_en.htm)