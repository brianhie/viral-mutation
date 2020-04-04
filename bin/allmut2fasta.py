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
elif sys.argv[1] == 'hiv':
    aligned_str = """
    -------MR-V---M---GIQRN----C------------Q--------------H----
    ------------L--------F--------R---------W-------------G----T
    -------M----------I------L---G-------M-I-----I----I-C-------
    -S--A--------------A-------E------N---------L------W-VTVY-YG
    V------P-VWKDA--------------------------------E--TTL-FCAS---
    -------------------------------DA-------KAYE----------TEKH--
    --NV-WATHACVPTD--PNPQEIHL----E-NVT-EEFNMW--------------KNNM-
    VEQ-MH-TDI-ISLW-DQSLKPCV------------KLT--PL--------C-------V
    ----------T-------L---------Q--------------C----------------
    ---T--------------------------------------------------------
    ------------------------------------------------------------
    -----------------------------------------N------------------
    ------------------------------------------------------------
    ------------V------------------------------T---------------N
    ------------------------------------------------------------
    ------------------N-----------------------------------------
    --------------I---------------------------------------------
    -------T----D--------------------------------------------D--
    ------------------------------------------------------------
    ------------------------------------------------------------
    --------------------------------M---------------------R-----
    -------G--------------------------------------E-------------
    -----L--------------------K-------NCSF---------N--M----T---T
    --------E----L------R-D-------K-K----Q--K--VYS-L-F------Y-R-
    ----L-------D---------V------V------------Q--------I--------
    ------------------N----------------E--------------------N---
    ------------------------------------Q-----------G-----------
    ------------------------N-----------------------------------
    --------------------R----------------------S-------NNSN-----
    --------K----E------------Y-------R----LIN--CNTSAI----TQACP-
    KVSF---E-P---IPI--H--YC-APAG------F-AI-L---KC-K-D-K-K-----F-
    N-----GT--G-P-CPSVSTVQCT--HGI--KPVVS-TQ-L-LL----------------
    ---NG-S--L-A-E------------E-E-----------V-MIR-SE-NITN-N---AK
    NIL--VQF--N---T--PVQ-IN---CTR---P-N-N-N---------------------
    T--R--K-S----I---R----I---G--P----G--Q-----A-F-----Y----A---
    ---T---G--D-----------------------------I------IG-D-I-------
    ----R-QA--HCNV---S-------K---A-TWN-E-TLGKV----VK------Q----L
    ----R-----K---------H---F--------GN-------------------N-----
    --T-----------I-------------------I-----R---F--A-----N--S--S
    ---G---GDLEV----TTH-SFNC-------GG--E-------FFY-------------C
    ---------------------N---------T------S--------G----------L-
    ----------F---------------------N---------------S-----------
    -------------T---------------W------------------------------
    --------------------IS-----N--------------T---------------S-
    --------------------------------VQ------------GS------------
    --N--------S--------------------T-------------------G-------
    ---------------S---------N---------------------D------------
    S----------------I-------T--------L-----P------C--RI-K----Q-
    ---II---------N-M-----W--QRIGQ----AMYAP---P-I------QG--VIRCV
    SNI----TG-LI--------L-------T-------R------D--------------G-
    --------G----------------S--------------------T-------------
    ------------------------------------------------------------
    ---N---------------S----------------T-------------T---------
    -----E-T----------F--RPGG-G--DMRDN---W-R------------SEL-----
    ------Y-KY------------KV-VK-IE-PL-G---V--APT-RAK-R---RV--V--
    --------------G-----R-----E---------K-------R-A---V------G--
    ----------I-G-A-V-FLG-F---L-GAA-GSTM-GAASMT-LTVQA-----------
    -------------------R-NLLS-G-I-VQQQS-N-LL-------------R-A-IE-
    AQQH--LLK-LTVWG-IKQL-----------Q-A---R-VLA----------V-ER-Y--
    ---LRDQQ-------------------------------LLGIW-----G-C-SGK----
    -LICTTNV-P-----W----N----S----S-W----S------------N----R----
    ----N----L----SE---IW---D--N--MTWLQW---DKEIS----NYTQI----IYG
    L-L----------------------EESQN-QQEKNE-Q-------DLLA----LD-KWA
    -------SLWNW-FD-ISNWLWYI-KIFIMIVGGLIGLRIVF-AV-L-SVI---HR--V-
    ------------R--QGY-S-PLSF-----Q------T---H-----T-----P------
    ---N---P-R--G----L-----DRPERI---EEED---GEQDRGR----STRLVS-GF-
    LA------------L-AWDD-LR--SLC-L--------FCYHRLRDFIL---IAAR-IVE
    ---L------------------L--------G-------HS-SLKG--LR-------L-G
    WE-G--------LKYLW--NL-------L-------AYW-----------------GR--
    -ELK---ISAIN-L-FDTI-AIAVA----EWTDRV-IEIGQ--------R------LC--
    --R---------A--FLH-I-P---RR-I-RQ--G---L-E-RALL----
    """.replace('\n', '').replace(' ', '')
    outfile = 'target/hiv/mutation/mutations_hiv.fa'
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
