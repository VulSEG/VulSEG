# encoding=utf-8
import os, sys
import glob
import argparse
from multiprocessing import Pool
from functools import partial
import subprocess


def get_all_file(path):
    path = path[0]
    file_list = []
    path_list = os.listdir(path)
    for path_tmp in path_list:
        full = path + path_tmp + '/'
        for file in os.listdir(full):
            file_list.append(file)
    return file_list


def parse_options():
    parser = argparse.ArgumentParser(description='Extracting Cpgs.')
    parser.add_argument('-i', '--input', help='The dir path of input', type=str, default='./novul_bin')
    parser.add_argument('-o', '--output', help='The dir path of output', type=str, default='./novul_output_pdg')
    parser.add_argument('-t', '--type', help='The type of procedures: parse or export', type=str, default='export')
    parser.add_argument('-r', '--repr', help='The type of representation: pdg or lineinfo_json', type=str,
                        default='pdg')
    args = parser.parse_args()
    return args


def joern_parse(file, outdir, inputdir):
    record_txt = os.path.join(outdir, "parse_res.txt")
    if not os.path.exists(record_txt):
        os.system("touch " + record_txt)
    with open(record_txt, 'r') as f:
        rec_list = f.readlines()
    name = file.split('/')[-1].split('.')[0]
    if name + '\n' in rec_list:
        print(" ====> has been processed: ", name)
        return
    print(' ----> now processing: ', name)
    out = outdir + name + '.bin'
    if os.path.exists(out):
        return
    os.environ['file'] = str(file)
    os.environ['out'] = str(out)  # parse后的文件名与source文件名称一致
    # os.system('sh joern-parse $file --language c --output $out')

    tmpStr = inputdir + file
    print(tmpStr)
    cmd = f'sh joern-parse {tmpStr} --language c --output {out}'

    print(cmd)

    os.system(cmd)


    with open(record_txt, 'a+') as f:
        f.writelines(name + '\n')


def joern_export(bin, outdir, repr):
    record_txt = os.path.join(outdir, "export_res.txt")
    if not os.path.exists(record_txt):
        os.system("touch " + record_txt)
    with open(record_txt, 'r') as f:
        rec_list = f.readlines()

    name = bin.split('/')[-1].split('.')[0]
    out = os.path.join(outdir, name)
    if name + '\n' in rec_list:
        print(" ====> has been processed: ", name)
        return
    print(' ----> now processing: ', name)
    input_path = "/Users/lumous/computer/ml/vulcnn-mac/data/sard/bins/Vul/"
    os.environ['bin'] = input_path + str(bin)
    os.environ['out'] = str(out)

    if repr != 'cpg':
        command = 'sh joern-export $bin' + " --repr " + str(repr) + ' --out $out'
        print(command)
        os.system(command)


    else:
        # 导出cpgs
        os.system('sh joern-export $bin' + " --repr=all --format=dot"  ' --out $out')




    len_outdir = len(glob.glob(outdir + '*'))
    print('--------------> len of outdir ', len_outdir)
    with open(record_txt, 'a+') as f:
        f.writelines(name + '\n')


def main():
    joern_path = './joern-cli/'
    os.chdir(joern_path)
    args = parse_options()
    type = args.type
    repr = args.repr

    input_path = args.input
    output_path = args.output

    if input_path[-1] == '/':
        input_path = input_path
    else:
        input_path += '/'

    if output_path[-1] == '/':
        output_path = output_path
    else:
        output_path += '/'

    pool_num = 16

    pool = Pool(pool_num)

    if type == 'parse':
        # files = get_all_file(input_path)
        # files = glob.glob(input_path + '*.c')

        all_files = os.listdir(input_path)
        # 筛选出以 .c 结尾的文件
        files = [file for file in all_files if file.endswith('.c')]

        pool.map(partial(joern_parse, outdir=output_path, inputdir=input_path), files)

    elif type == 'export':
        # bins = glob.glob(input_path + '*.bin')
        all_files = os.listdir(input_path)
        # 筛选出以 .c 结尾的文件
        bins = [file for file in all_files if file.endswith('.bin')]
        try:

            if repr == 'pdg':
                pool.map(partial(joern_export, outdir=output_path, repr=repr), bins)
            else:
                pool.map(partial(joern_export, outdir=output_path, repr=repr), bins)
        except Exception as e:
            print(e)

    else:
        print('Type error!')


if __name__ == '__main__':
    main()