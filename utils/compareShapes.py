#!/usr/bin/env python
# %%
import sys,os
import uproot
import numpy as np

# %%
p1=sys.argv[1]
p2=sys.argv[2]
print(sys.argv)
shape1 = uproot.open(p1)
shape2 = uproot.open(p2)

GREEN='\x1b[6;30;42m'
RED='\x1b[0;30;41m'
END='\x1b[0m'

# %%
print("{} | {}".format(p1,p2))
print(GREEN+">"+END+": key in {} but not in {} ".format(p1,p2))
print(RED+"<"+END+": key in {} but not in {} ".format(p2,p1))

keyset1=shape1.allkeys()
keyset2=shape2.allkeys()
ignoreS={"CMS_htt_dyShape"}
ignoreS= ignoreS | {"CMS_scale_mc_t_1prong", "CMS_scale_mc_t_3prong"} # mc_tau energy scale added
# ignoreS= ignoreS | {"CMS_eff_t_dm11","CMS_eff_emb_t_dm11"} ##dm11 fixed
# ignoreS= ignoreS | {"Down;"} # ignore down shifts
#{"CMS_eff_emb_t_","CMS_eff_t_","CMS_scale_mc_t_"}

keyset1 = {key.replace("data_obs","data") for key in keyset1 if not any([pattern in key for pattern in ignoreS]) }
keyset2 = {key.replace("data_obs","data") for key in keyset2 if not any([pattern in key for pattern in ignoreS]) }

diffkeys = sorted(set(keyset1) ^ set(keyset2))

if len(diffkeys)==0:
    print("Keys are the same")
else:
    for key in diffkeys:
        if key in keyset1:
            print(GREEN+">"+END+" {key} >".format(key=key))
        else:
            print(RED+"<"+END+" {key} <".format(key=key))

# %%
exit(0)
keyset1=shape1.allkeys()
keyset2=shape2.allkeys()
samekeys = sorted(set(keyset1) & set(keyset2))

print("Analysing Histogramm content")
for key in samekeys:
    arr1,bin1=shape1[key].allnumpy()
    arr2,bin2=shape2[key].allnumpy()
    arr1=np.round(arr1,decimals=4)
    arr2=np.round(arr2,decimals=4)
    if not np.array_equal(arr1,arr2) or not np.array_equal(bin1,bin2):
        if not np.array_equal(bin1,bin2):
            print("{} \n binning different: \n {} \n vs. \n {}".format(key, bin1, bin2))
        else:
            print("{} \n contents different: \n {} \n vs. \n {}".format(key, arr1, arr2))


# %%