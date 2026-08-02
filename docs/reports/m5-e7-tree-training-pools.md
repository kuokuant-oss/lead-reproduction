# M5 E7 Tree Training Pools

The frozen five-fold manifest assigns every one of the 725 even buildings to
exactly one validation fold. Steam validation row counts are 277,139; 272,382;
270,145; 268,357; and 270,081. Corresponding steam anomaly counts are 18,472;
16,677; 15,865; 13,504; and 12,453.

For each fold and support cell, the matching neutral pool is drawn from the
full-support pool with fixed pair seeds: `n00` 110/120, `n01` 210/220, `n10`
310/320, `n11` 410/420. The manifest records row-order digests and enforces
exact equality of total, positive, and negative row counts for every pair.

No odd-building labels were used in this manifest construction.
