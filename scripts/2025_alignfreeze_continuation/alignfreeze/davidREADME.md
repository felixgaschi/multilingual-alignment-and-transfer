## New strategies

The following regular expression for the strategy represent the number of intervals to switch up the layer to be finetuned.
For example, if n_epochs=3 and the provided number is `3` for topdown, then every epoch, num_layers // `3` layers will be chosen first to be unfrozen, and so on.
- before_gradual_random_[0-9+] <---- the name here is misleading for now... but yeah essentially it's jsut oneatatime (see later)
- before_gradual_bottomup_[0-9+]
- before_gradual_topdown_[0-9+]
Note that this will have gradual unfreezing

To have one at a time unfreezing, use one at a time:
- before_oneatatime_random_[0-9+]
- before_oneatatime_bottomup_[0-9+]
- before_oneatatime_topdown_[0-9+]

If you want to select `x` number of layers initially to be frozen, then use this regex:
- before_random_realign_[0-9]+

THis will freeze layer per block (block defined by num_layers // `x`)

