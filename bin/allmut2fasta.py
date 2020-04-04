import sys

if sys.argv[1] == 'h3':
    aligned_str = """
    ---------M------KT----II--A-L--SYI-LCLVFA--Q---KLPG-ND-NSTAT
    LCLGHH---AVPNGTIVKTITNDQIEVTNATELVQSSSTGEICD-SPHQILDGKNCTLID
    AL--LGDPQCDGFQN---KKWDLFVER--------------SKAYSNCYP----------
    ----------YDVPDYASLRSLVASSGTLEFN-NESF----NW-TGVTQNGT-SSACI-R
    -R-S-KNSFFSRL--NWLT----H----LNFKYPALNVTM--PNNE------QFDKLYIW
    GVLHPGTDKDQ---IFLYAQASGRIT-VSTKRSQQTVSPNIGSR-PRVRNIPSRISIYWT
    IVKPGDILLINSTGNLIAPRGYFKIRS-----------GKSSIMRS-DAPIGK-CNSECI
    TPNGSIP---------NDKPFQNVNRITYGACPRYVKQN--TLKLATGM--------RNV
    P-E---------------K-----------Q-----T------------RGIFGAIAGFI
    ENGWEGMVDGWYGFRHQNSEGRGQAADLKSTQ--AAIDQINGKLNRLIGKTNEKFHQIEK
    EFSEVEGRIQDLEKYVEDTKIDLWSYNAELLVALENQHTIDLTDSEMNKLFEKTKKQ-LR
    ENAEDMGNGCFKIYHKCDNACIGSIRNGTYDHDVYRDEALNNRFQIKGVELKS---GYKD
    WILWISFAI-------SCFLLCVALLGFIMWACQ--KG------N---IRCN--I-CI--
    ----------------
    """.replace('\n', '').replace(' ', '')
    outfile = 'target/flu/mutation/mutations_h3.fa'
elif sys.argv[1] == 'h1':
    aligned_str = """
    ---------M------K--A-KLL----V----L-LYAFVA--T---D----AD-----T
    ICIGYH---ANNSTDTVDTIFEKNVAVTHSVNLLEDRHNGKLCKLKGIAPLQLGKCNITG
    WL--LGNPECDSLLPA--RSWSYIVETP-------------NSENGACYP----------
    ----------GDFIDYEELREQLSSVSSLERF--EIFPKESSWPNH-TFNGV-TVSCS-H
    -R-G-KSSFYRNL--LWLT-K--K--G--D-SYPKLTNSY--VNNK------GKEVLVLW
    GVHHPSSSDEQ---QSLYSNGNAYVS-VASSNYNRRFTPEIAAR-PKVKDQHGRMNYYWT
    LLEPGDTIIFEATGNLIAPWYAFALSR--G--------FESGIITS-NASMHE-CNTKCQ
    TPQGSIN---------SNLPFQNIHPVTIGECPKYVRST--KLRMVTGL--------RNI
    P-S-------I-------------------Q-----Y------------RGLFGAIAGFI
    EGGWTGMIDGWYGYHHQNEQGSGYAADQKSTQ--NAINGITNKVNSVIEKMNTQFTAVGK
    EFNNLEKRMENLNKKVDDGFLDIWTYNAELLVLLENERTLDFHDLNVKNLYEKVKSQ-LK
    NNAKEIGNGCFEFYHKCDNECMESVRNGTYDYPKYSEESKLNREKIDGVKLESMG-VYQI
    LAIYSTVAS--------SLVLLVSLGAISFWMCS--NG------S---LQCR--I-CI--
    ----------------
    """.replace('\n', '').replace(' ', '')
    outfile = 'target/flu/mutation/mutations_h1.fa'
else:
    raise ValueError('invalid option {}'.format(sys.argv[1]))

AAs = [
    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H',
    'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W',
    'Y', 'V', 'X', 'Z', 'J', 'U', 'B',
]

with open(outfile, 'w') as of:
    for i in range(len(aligned_str)):
        if aligned_str[i] == '-':
            continue
        for aa in AAs:
            if aligned_str[i] == aa:
                continue
            name = 'mut_{}_{}'.format(i, aa)
            mutable = aligned_str[:i] + aa + aligned_str[i + 1:]
            of.write('>{}\n'.format(name))
            of.write('{}\n'.format(mutable))
