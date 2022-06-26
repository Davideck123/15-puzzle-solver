# 15 Puzzle solver

The search uses IDA* algorithm with walking distance heuristic - you have to run the following command at the start
```prolog
create_wd_database.
```
to generate all 24964 possible boards for this heuristic (otherwise it won't work) - it shouldn't take a bit more than a minute on a decent computer.

You can also use a much weaker heuristic (which doesn't have to generate anything) `MAX(inversion distance, manhattan)` but you have to change the `heuristic` in the code.

```prolog
% Use this heuristic if you don't want to generate boards for walking distance heuristic

%heuristic(Current, _, Res) :-
%    manhattan(Current, M),
%    inversion_distance(Current, I),
%    Res is max(M, I).


heuristic(Current, _, Res) :-
    get_walking_distance(Current, Res).
```

The puzzle board is represented as one list of numbers 0-15 left to right, top to bottom, eg. the final state is represented as [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,0].
Then you can simply run (for example)
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
- `test1` - 35 moves
- `test2` - 36 moves
- `test3` - 37 moves
- `test4` - 40 moves
- `test5` - 41 moves
- `test6` - 46 moves

Each test shouldn't take more than a couple of minutes at max

### Interesting and helpful information about Walking distance and 15 puzzle in general:

- [https://michael.kim/blog/puzzle](https://michael.kim/blog/puzzle)
- [http://kociemba.org/themen/fifteen/fifteensolver.html](http://kociemba.org/themen/fifteen/fifteensolver.html)
- [https://web.archive.org/web/20141224035932/http://juropollo.xe0.ru/stp_wd_translation_en.htm](https://web.archive.org/web/20141224035932/http://juropollo.xe0.ru/stp_wd_translation_en.htm)
