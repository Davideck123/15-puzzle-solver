
% solvable(+Puzzle) - succeeds if Puzzle is solvable 
% (considering the standard configuration [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,0])
solvable(Puzzle) :- 
    sum_row_inversions(Puzzle, R),
    empty_square_row(Puzzle, I),
    0 is (R + I) mod 2.


% empty_square_row(+Puzzle, ?Index) - Index (1-based) of the row with empty square
empty_square_row(Puzzle, Index) :-
    nth0(I, Puzzle, 0),
    Index is I div 4 + 1,
    !.

pred(X, Y) :- 
    Y < X,
    Y > 0.

% sum_row_inversions(+Board, -Count) - Count is the number of inversions for particular Board
sum_row_inversions([], 0).
sum_row_inversions([X|Xs], R) :-
    include(pred(X), Xs, Ys),
    length(Ys, InversionCount),
    sum_row_inversions(Xs, Sum),
    R is InversionCount + Sum.

% transpose_(+X, -XT) - rows of X are columns of XT
% [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,0] -> [1,5,9,13,2,6,10,14,3,7,11,15,4,8,12,0]
transpose_([A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P], [A, E, I, M, B, F, J, N, C, G, K, O, D, H, L, P]).

pred2(X, Y) :-
    Config = [1,5,9,13,2,6,10,14,3,7,11,15,4,8,12,0],
    nth0(IndexX, Config, X),
    nth0(IndexY, Config, Y),
    IndexX > IndexY,
    X \= 0,
    Y \= 0.

% sum_column_inversions(+Board, -Count) - Count is the number of inversions for particular Board
sum_column_inversions([], 0).
sum_column_inversions([X|Xs], R) :-
    include(pred2(X), Xs, Ys),
    length(Ys, InversionCount),
    sum_column_inversions(Xs, Sum),
    R is InversionCount + Sum.
    

% inversion_distance(+Board, -N) - N is the result of inversion distance heuristic for Board
inversion_distance(State, Result) :-
    sum_row_inversions(State, V),
    divmod(V, 3, X, Y),
    Vertical is X + Y,
    transpose_(State, StateT),
    sum_column_inversions(StateT, H),
    divmod(H, 3, X2, Y2),
    Horizontal is X2 + Y2,
    Result is Vertical + Horizontal.
    
% index(+I, ?X, ?Y) - convert I-th index to (X, Y) index in 4x4 square
index(I, X, Y) :- divmod(I, 4, X, Y).  


% manhattan(+Board, -N) - N is the result of manhattan heuristic for Board
manhattan(Current, Res) :-
    manhattan(Current, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,0], 15, Res).

manhattan(_, _, 0, 0).
manhattan(Current, Final, Num, Sum) :-
    nth0(Index, Current, Num),
    nth0(FinalIndex, Final, Num),
    index(Index, X1, Y1),
    index(FinalIndex, X2, Y2),
    Num1 is Num - 1,
    manhattan(Current, Final, Num1, Sum1),
    Sum is abs(X1 - X2) + abs(Y1 - Y2) + Sum1.


% Use this heuristic if you don't want to generate boards for walking distance heuristic

%heuristic(Current, _, Res) :-
%    manhattan(Current, M),
%    inversion_distance(Current, I),
%    Res is max(M, I).


heuristic(Current, _, Res) :-
    get_walking_distance(Current, Res).


solve_astar(Current, Final, Limit, Moves) :-
    solve_astar(Current, Final, [Current], [], Limit, Moves).

solve_astar(Final, Final, _, MovesBackwards, Limit, Moves) :-
    Limit >= 0,
    reverse(MovesBackwards, Moves).

solve_astar(Current, Final, StateAcc, MovesBackwards, Limit, Moves) :-
    heuristic(Current, Final, H),
    Limit >= H,
    L1 is Limit - 1,
    move(Current, NewState, Direction),
    \+member(NewState, StateAcc),
    solve_astar(NewState, Final, [NewState|StateAcc], [Direction|MovesBackwards], L1, Moves).

% solve_astar(+Puzzle, +L, -Moves) - A* search to the final state, use at most 'L' Moves 
solve_astar(Current, Limit, Moves) :-
    solve_astar(Current, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,0], Limit, Moves).

% solve_idastar(+Current, +Final, -Moves) - IDA* search from Current to Final
% 80 is the max number of moves to solve a solvable Puzzle
solve_idastar(Current, Final, Moves):-
    heuristic(Current, Final, H),
    between(H, 80, Limit),
    solve_astar(Current, Final, Limit, Moves).

% solve(+Board, -Moves) - IDA* search to the final state, check solvability first
solve(Board, Moves) :-
    solvable(Board),
    solve_idastar(Board, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,0], Moves).


% TESTS

% [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,0]

% [15,2,1,12,8,5,6,11,4,9,10,0,3,14,13,7] - unsolvable
test0 :- solve([15,2,1,12,8,5,6,11,4,9,10,0,3,14,13,7], _).

% [1,10,15,4,13,6,3,8,2,9,12,7,14,5,0,11] - 35 moves
test1 :- 
    solve([1,10,15,4,13,6,3,8,2,9,12,7,14,5,0,11], M),
    write(M),
    length(M, L),
    write(" Moves: "),
    writeln(L).

% [1,10,15,4,13,0,6,8,2,9,3,7,14,5,12,11] - 36 moves
test2 :- 
    solve([1,10,15,4,13,0,6,8,2,9,3,7,14,5,12,11], M),
    write(M),
    length(M, L),
    write(" Moves: "),
    writeln(L). 

% [1,10,15,4,0,13,6,8,2,9,3,7,14,5,12,11] - 37 moves
test3 :- 
    solve([1,10,15,4,0,13,6,8,2,9,3,7,14,5,12,11], M),
    write(M),
    length(M, L),
    write(" Moves: "),
    writeln(L).

% [3,10,2,8,9,6,7,0,5,4,15,1,13,14,12,11] - 40 moves
test4 :- 
    solve([3,10,2,8,9,6,7,0,5,4,15,1,13,14,12,11], M),
    write(M),
    length(M, L),
    write(" Moves: "),
    writeln(L).

% [7,9,1,3,12,2,5,4,11,0,15,8,14,6,13,10] - 41 moves
test5 :- 
    solve([7,9,1,3,12,2,5,4,11,0,15,8,14,6,13,10], M),
    write(M),
    length(M, L),
    write(" Moves: "),
    writeln(L).

% [1,12,7,8,11,14,10,0,2,6,4,5,9,13,3,15] - 46 moves
test6 :- 
    solve([1,12,7,8,11,14,10,0,2,6,4,5,9,13,3,15], M),
    write(M),
    length(M, L),
    write(" Moves: "),
    writeln(L).

% [3,15,10,13,0,9,12,11,6,5,8,1,4,2,14,7] - 63 moves

% [13,10,11,6,5,7,4,8,1,12,14,9,3,15,2,0] unsolvable

%%%%%%%%%%%%%%%%%%%%%%% EMPTY SQUARE MOVES %%%%%%%%%%%%%%%%%%%%%%%

% move empty square to the right
move([0, X|Xs], [X, 0|Xs], r).
move([A, 0, X|Xs], [A, X, 0|Xs], r).
move([A, B, 0, X|Xs], [A, B, X, 0|Xs], r).
move([A, B, C, D, 0, X|Xs], [A, B, C, D, X, 0|Xs], r).
move([A, B, C, D, E, 0, X|Xs], [A, B, C, D, E, X, 0|Xs], r).
move([A, B, C, D, E, F, 0, X|Xs], [A, B, C, D, E, F, X, 0|Xs], r).
move([A, B, C, D, E, F, G, H, 0, X|Xs], [A, B, C, D, E, F, G, H, X, 0|Xs], r).
move([A, B, C, D, E, F, G, H, I, 0, X|Xs], [A, B, C, D, E, F, G, H, I, X, 0|Xs], r).
move([A, B, C, D, E, F, G, H, I, J, 0, X|Xs], [A, B, C, D, E, F, G, H, I, J, X, 0|Xs], r).
move([A, B, C, D, E, F, G, H, I, J, K, L, 0, X|Xs], [A, B, C, D, E, F, G, H, I, J, K, L, X, 0|Xs], r).
move([A, B, C, D, E, F, G, H, I, J, K, L, M, 0, X|Xs], [A, B, C, D, E, F, G, H, I, J, K, L, M, X, 0|Xs], r).
move([A, B, C, D, E, F, G, H, I, J, K, L, M, N, 0, X], [A, B, C, D, E, F, G, H, I, J, K, L, M, N, X, 0], r).

% move empty square to the left
move([A, B, C, D, E, F, G, H, I, J, K, L, M, N, X, 0], [A, B, C, D, E, F, G, H, I, J, K, L, M, N, 0, X], l).
move([A, B, C, D, E, F, G, H, I, J, K, L, M, X, 0|Xs], [A, B, C, D, E, F, G, H, I, J, K, L, M, 0, X|Xs], l).
move([A, B, C, D, E, F, G, H, I, J, K, L, X, 0|Xs], [A, B, C, D, E, F, G, H, I, J, K, L, 0, X|Xs], l).
move([A, B, C, D, E, F, G, H, I, J, X, 0|Xs], [A, B, C, D, E, F, G, H, I, J, 0, X|Xs], l).
move([A, B, C, D, E, F, G, H, I, X, 0|Xs], [A, B, C, D, E, F, G, H, I, 0, X|Xs], l).
move([A, B, C, D, E, F, G, H, X, 0|Xs], [A, B, C, D, E, F, G, H, 0, X|Xs], l).
move([A, B, C, D, E, F, X, 0|Xs], [A, B, C, D, E, F, 0, X|Xs], l).
move([A, B, C, D, E, X, 0|Xs], [A, B, C, D, E, 0, X|Xs], l).
move([A, B, C, D, X, 0|Xs], [A, B, C, D, 0, X|Xs], l).
move([A, B, X, 0|Xs], [A, B, 0, X|Xs], l).
move([A, X, 0|Xs], [A, 0, X|Xs], l).
move([X, 0|Xs], [0, X|Xs], l).

% move empty square down
move([0, B, C, D, 
       X, F, G, H, 
       I, J, K, L, 
       M, N, O, P], 
      [X, B, C, D, 
       0, F, G, H, 
       I, J, K, L, 
       M, N, O, P], d).

move([A, 0, C, D, 
       E, X, G, H, 
       I, J, K, L, 
       M, N, O, P], 
      [A, X, C, D, 
       E, 0, G, H, 
       I, J, K, L, 
       M, N, O, P], d).

move([A, B, 0, D, 
       E, F, X, H, 
       I, J, K, L, 
       M, N, O, P], 
      [A, B, X, D, 
       E, F, 0, H, 
       I, J, K, L, 
       M, N, O, P], d).

move([A, B, C, 0, 
       E, F, G, X, 
       I, J, K, L, 
       M, N, O, P], 
      [A, B, C, X, 
       E, F, G, 0, 
       I, J, K, L, 
       M, N, O, P], d).

move([A, B, C, D, 
       0, F, G, H, 
       X, J, K, L, 
       M, N, O, P], 
      [A, B, C, D, 
       X, F, G, H, 
       0, J, K, L, 
       M, N, O, P], d).

move([A, B, C, D, 
       E, 0, G, H, 
       I, X, K, L, 
       M, N, O, P], 
      [A, B, C, D, 
       E, X, G, H, 
       I, 0, K, L, 
       M, N, O, P], d).

move([A, B, C, D, 
       E, F, 0, H, 
       I, J, X, L, 
       M, N, O, P], 
      [A, B, C, D, 
       E, F, X, H, 
       I, J, 0, L, 
       M, N, O, P], d).

move([A, B, C, D, 
       E, F, G, 0, 
       I, J, K, X, 
       M, N, O, P], 
      [A, B, C, D, 
       E, F, G, X, 
       I, J, K, 0, 
       M, N, O, P], d).

move([A, B, C, D, 
       E, F, G, H, 
       0, J, K, L, 
       X, N, O, P], 
      [A, B, C, D, 
       E, F, G, H, 
       X, J, K, L, 
       0, N, O, P], d).

move([A, B, C, D, 
       E, F, G, H, 
       I, 0, K, L, 
       M, X, O, P], 
      [A, B, C, D, 
       E, F, G, H, 
       I, X, K, L, 
       M, 0, O, P], d).

move([A, B, C, D, 
       E, F, G, H, 
       I, J, 0, L, 
       M, N, X, P], 
      [A, B, C, D, 
       E, F, G, H, 
       I, J, X, L, 
       M, N, 0, P], d).

move([A, B, C, D, 
       E, F, G, H, 
       I, J, K, 0, 
       M, N, O, X], 
      [A, B, C, D, 
       E, F, G, H, 
       I, J, K, X, 
       M, N, O, 0], d).

% move empty square up
move([X, B, C, D, 
       0, F, G, H, 
       I, J, K, L, 
       M, N, O, P], 
      [0, B, C, D, 
       X, F, G, H, 
       I, J, K, L, 
       M, N, O, P], u).

move([A, X, C, D, 
       E, 0, G, H, 
       I, J, K, L, 
       M, N, O, P], 
      [A, 0, C, D, 
       E, X, G, H, 
       I, J, K, L, 
       M, N, O, P], u).

move([A, B, X, D, 
       E, F, 0, H, 
       I, J, K, L, 
       M, N, O, P], 
      [A, B, 0, D, 
       E, F, X, H, 
       I, J, K, L, 
       M, N, O, P], u).

move([A, B, C, X, 
       E, F, G, 0, 
       I, J, K, L, 
       M, N, O, P], 
      [A, B, C, 0, 
       E, F, G, X, 
       I, J, K, L, 
       M, N, O, P], u).

move([A, B, C, D, 
       X, F, G, H, 
       0, J, K, L, 
       M, N, O, P], 
      [A, B, C, D, 
       0, F, G, H, 
       X, J, K, L, 
       M, N, O, P], u).

move([A, B, C, D, 
       E, X, G, H, 
       I, 0, K, L, 
       M, N, O, P], 
      [A, B, C, D, 
       E, 0, G, H, 
       I, X, K, L, 
       M, N, O, P], u).

move([A, B, C, D, 
       E, F, X, H, 
       I, J, 0, L, 
       M, N, O, P], 
      [A, B, C, D, 
       E, F, 0, H, 
       I, J, X, L, 
       M, N, O, P], u).

move([A, B, C, D, 
       E, F, G, X, 
       I, J, K, 0, 
       M, N, O, P], 
      [A, B, C, D, 
       E, F, G, 0, 
       I, J, K, X, 
       M, N, O, P], u).

move([A, B, C, D, 
       E, F, G, H, 
       X, J, K, L, 
       0, N, O, P], 
      [A, B, C, D, 
       E, F, G, H, 
       0, J, K, L, 
       X, N, O, P], u).

move([A, B, C, D, 
       E, F, G, H, 
       I, X, K, L, 
       M, 0, O, P], 
      [A, B, C, D, 
       E, F, G, H, 
       I, 0, K, L, 
       M, X, O, P], u).

move([A, B, C, D, 
       E, F, G, H, 
       I, J, X, L, 
       M, N, 0, P], 
      [A, B, C, D, 
       E, F, G, H, 
       I, J, 0, L, 
       M, N, X, P], u).

move([A, B, C, D, 
       E, F, G, H, 
       I, J, K, X, 
       M, N, O, 0], 
      [A, B, C, D, 
       E, F, G, H, 
       I, J, K, 0, 
       M, N, O, X], u).

%%%%%%%%%%%%%%%%%%%% WALKING DISTANCE HEURISTIC %%%%%%%%%%%%%%%%%%%%

:- dynamic walking_distance/2.
:- dynamic visited/1.

% generate all 24964 possible boards for walking distance heuristic
create_wd_database :-
    Start = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,0],
    list_to_matrix(Start, M),
    maplist(sort, M, SM),
    retractall(walking_distance(_, _)),
    retractall(visited(_)),
    bfs_wd([(SM, 0)]).

% bfs search
bfs_wd([]).
bfs_wd([(X, L)|Qs]) :-
    get_row_pattern(X, P),
    \+visited(P),
    !,
    assert(walking_distance(P, L)),
    assert(visited(P)),
    findall((X1, L1), (wd_move(X, X1), get_row_pattern(X1, P1), \+visited(P1), L1 is L + 1), Ss),
    %findall((X1, L1), (wd_move(X, X1), get_row_pattern(X1, P1), \+walking_distance(P1, _), L1 is L + 1), Ss),
    append(Qs, Ss, F),
    %writeln(L),
    bfs_wd(F).


bfs_wd([_|Qs]) :-
    !,
    bfs_wd(Qs).

% need transpose function
:- use_module(library(clpfd)).

% [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,0] -> [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 0]]
list_to_matrix([], []).
list_to_matrix([A, B, C, D|Xs], [[A, B, C, D]|Ys]) :-
    list_to_matrix(Xs, Ys).

% walking distance moves

% move from the 1st row/column
wd_move([[0|As], Bs, Cs, Ds], [[E|As], [0|Xs], Cs, Ds]) :-
    select(E, Bs, Xs).

% move from the 2nd row/column
wd_move([As, [0|Bs], Cs, Ds], [[0|Xs], [E|Bs], Cs, Ds]) :-
    select(E, As, Xs).

wd_move([As, [0|Bs], Cs, Ds], [As, [E|Bs], [0|Xs], Ds]) :-
    select(E, Cs, Xs).
    
% move from the 3rd row/column
wd_move([As, Bs, [0|Cs], Ds], [As, [0|Xs], [E|Cs], Ds]) :-
    select(E, Bs, Xs).

% move from the 4th row/column
wd_move([As, Bs, [0|Cs], Ds], [As, Bs, [E|Cs], [0|Xs]]) :-
    select(E, Ds, Xs).

wd_move([As, Bs, Cs, [0|Ds]], [As, Bs, [0|Xs], [E|Ds]]) :-
    select(E, Cs, Xs).


% plus_one(+I, +Xs, -Ys) - add one to the I-th index of Xs
plus_one(I, Xs, Ys) :-
    nth0(I, Xs, E, R),
    E1 is E + 1,
    nth0(I, Ys, E1, R).
    

pattern(_, [], [0, 0, 0, 0]).
pattern(Op, [0|Xs], Ss) :-
    !,
    pattern(Op, Xs, Ss).
pattern(Op, [X|Xs], Zs) :-
    pattern(Op, Xs, Ss),
    Expr =..[Op, (X - 1), 4],
    I is Expr,
    plus_one(I, Ss, Zs).


row_pattern(L, P) :- 
    pattern(div, L, P),
    !.

column_pattern(L, P) :- 
    pattern(mod, L, P),
    !.

% get_row_pattern(+Puzzle, -P) - get walking distance row pattern P for the Puzzle in the matrix form
% [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,0]] -> [[4, 0, 0, 0], [0, 4, 0, 0], [0, 0, 4, 0], [0, 0, 0, 3]]
get_row_pattern(Matrix, P) :-
    maplist(row_pattern, Matrix, P).

% get_column_pattern(+Puzzle, -P) - get walking distance column pattern P for the Puzzle in the matrix form
% [[1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15], [4, 8, 12, 0]] -> [[4, 0, 0, 0], [0, 4, 0, 0], [0, 0, 4, 0], [0, 0, 0, 3]]
get_column_pattern(Matrix, P) :-
    maplist(column_pattern, Matrix, P).


% get_vertical(+Matrix, -D) - D is the vertical walking distance for Matrix
get_vertical(M, D) :-
    get_row_pattern(M, P),
    walking_distance(P, D).

% get_horizontal(+Matrix, -D) - D is the horizontal walking distance for Matrix
get_horizontal(M, D) :-
    transpose(M, MT),
    get_column_pattern(MT, P),
    walking_distance(P, D).

% get_walking_distance(+Puzzle, -N) - N is the result of walking distance heuristic for Puzzle
get_walking_distance(L, D) :-
    list_to_matrix(L, M),
    get_vertical(M, V),
    get_horizontal(M, H),
    D is V + H,
    !.
