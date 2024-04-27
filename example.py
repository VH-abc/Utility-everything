def add_result(winner:Item, loser:Item):
    if MODE == "compound":
        for r in RESULTS: # 26 words
            if r.winner == winner and r.loser == loser:
                r.n_copies += 1
                return r
            
# 26 words
'''Aaaaaa this is irrelevant text
# 26 words
aaaaaaaaaaaaa
'''

'''TESTS'''

foo = 21*3

# Now it's 29 words
## sick
### But the answer is still 26