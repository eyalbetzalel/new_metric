# imports 

import argparse
import os
import pandas as pd
import argparse

from clean_fid.cleanfid import fid  

# import torch
# from swd_pytorch.swd import swd
# from torchvision import datasets

from FID_IS_infinity.score_infinity import calculate_FID_infinity_path, calculate_IS_infinity_path

def main(
    path_source,
    path_test,
    batch_size,
    output_path,
):
    # Run clean-fid + other in this suite
    
    cleanfid_Score = fid.compute_fid(path_source, path_test)
    kid_score = fid.compute_kid(path_source, path_test)

    # Run SWD 
    
    # TODO images to tensors
    # data_source = datasets.ImageFolder(root=path_source)
    # data_test = datasets.ImageFolder(root=path_test)

    # data_source_loader = torch.utils.data.DataLoader(data_source, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4)
    # data_test_loader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4)
    # for 
    # swd_score = swd(x1, x2, device="cuda") # Fast estimation if device="cuda"

    # Run inf FID / IS :

    IS_infinity = calculate_IS_infinity_path(path_test, batch_size)
    FID_infinity = calculate_FID_infinity_path(path_source, path_test, batch_size)
    
    # Output :

    model = path_test.split("/")[-1]
    text_file_name = model + ".txt"
    out_path = os.join.path(output_path, text_file_name)
    text_file = open(out_path, "w")
    text_file.write(f"IS inf : {IS_infinity}\n")
    text_file.write(f"FID inf : {FID_infinity}\n")
    text_file.write(f"Clean FID : {cleanfid_Score}\n")
    text_file.write(f"KID : {kid_score}\n")
    # text_file.write(f"SWD : {swd_score}\n")
    text_file.close()

if __name__ == "__main__":
    
    # WP-GAN : 

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_source", type=str, default="./", help="path to original distribution dataset")
    parser.add_argument("--path_test", type=str, default="./", help="path to expiremntal distribution dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size for inf fid")
    parser.add_argument("--output_path", type=str, default="./", help="path to results savind directory")
    opt = parser.parse_args()
    kwargs = vars(opt)
    print(opt)
    main(**kwargs)





