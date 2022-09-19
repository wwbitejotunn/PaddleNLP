#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_ihfft2
"""

from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestFFtihfft2(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float64, np.float32]
        # self.debug = True
        # self.static = True
        # enable check grad
        self.enable_backward = False
        # self.delta = 1e-5


obj = TestFFtihfft2(paddle.fft.ihfft2)


@pytest.mark.api_fft_ihfft2_vartype
def test_ihfft2_base():
    """
    base
    """
    np.random.seed(33)
    x = np.random.rand(3, 4) * 4 - 2
    res = np.array(
        [
            [(-0.02548679830542297 + 0j), (0.1345758574077266 - 0.16178232891633898j), (0.15547724417759462 + 0j)],
            [
                (-0.3023935796613055 - 0.1435048601502342j),
                (0.10016192509381111 + 0.5034648768636009j),
                (-0.10315071284050961 - 0.29263737512305726j),
            ],
            [
                (-0.3023935796613055 + 0.1435048601502342j),
                (-0.39716845806034856 - 0.1520068177833663j),
                (-0.10315071284050961 + 0.29263737512305726j),
            ],
        ]
    )
    obj.base(res=res, x=x)


@pytest.mark.api_fft_ihfft2_parameters
def test_ihfft2_0():
    """
    default
    x: tensor-3d
    """
    np.random.seed(33)
    x_data = np.random.rand(3, 4, 5) * 4 - 2
    res = np.array(
        [
            [
                [
                    (-0.05435021182450783 + 0j),
                    (0.11848687151888815 - 0.01464927288956578j),
                    (0.09956008214310907 - 0.07337608601880574j),
                ],
                [
                    (0.05741920268510847 - 0.1770233724221432j),
                    (0.014350887517698724 - 0.4041810768116577j),
                    (-0.11242156583807675 - 0.1371724605314702j),
                ],
                [
                    (-0.2683908089705767 + 0j),
                    (0.2423067399606768 - 0.08637872653447728j),
                    (-0.08544623592144776 + 0.074660554772907j),
                ],
                [
                    (0.05741920268510847 + 0.1770233724221432j),
                    (-0.284358681190064 + 0.25617002766964203j),
                    (-0.3915065356028906 - 0.17642004356292046j),
                ],
            ],
            [
                [
                    (-0.14210655566913039 + 0j),
                    (0.21717119946013935 + 0.0878614953164668j),
                    (-0.2204106223379333 + 0.053469712054527246j),
                ],
                [
                    (-0.21396379713374436 + 0.09527538941527554j),
                    (-0.12889294954237768 - 0.06035159088716653j),
                    (-0.07748887296393812 - 0.13641387308288438j),
                ],
                [
                    (0.11807429688229666 + 0j),
                    (0.13447373758178877 + 0.2667763854075799j),
                    (0.5719483825581403 - 0.18989885038671078j),
                ],
                [
                    (-0.21396379713374436 - 0.09527538941527554j),
                    (-0.030331770622540624 - 0.032971741556724374j),
                    (0.20126451940167495 - 0.07070576118887008j),
                ],
            ],
            [
                [
                    (-0.4112573532075301 + 0j),
                    (0.06128815795358879 + 0.2961216152873792j),
                    (0.18521058251115924 + 0.005251529058716814j),
                ],
                [
                    (-0.040475885854851124 + 0.09443612771184898j),
                    (0.03990374416202082 + 0.09110395717672135j),
                    (-0.15611009211852006 - 0.06803780751948094j),
                ],
                [
                    (0.02650780389798904 + 0j),
                    (-0.21163787292223024 - 0.07634126991726414j),
                    (0.11581163625275553 - 0.38510659075138753j),
                ],
                [
                    (-0.040475885854851124 - 0.09443612771184898j),
                    (0.12122677495698549 - 0.09185061001198253j),
                    (-0.19771825145879346 - 0.13035795924856106j),
                ],
            ],
        ]
    )
    obj.run(res=res, x=x_data)


@pytest.mark.api_fft_ihfft2_parameters
def test_ihfft2_1():
    """
    default
    x: tensor-4d
    """
    np.random.seed(33)
    x_data = np.random.rand(4, 3, 3, 3) * 4 - 2
    res = np.array(
        [
            [
                [
                    [(-0.18732181353528182 + 0j), (-0.5540249307292431 + 0.3838192281097648j)],
                    [(-0.1667215255756483 - 0.12995344547386195j), (0.01819563083653894 + 0.034536323262227026j)],
                    [(-0.1667215255756483 + 0.12995344547386195j), (0.29323198711145326 - 0.3732822568893155j)],
                ],
                [
                    [(0.13220991875760155 + 0j), (-0.03322994679792096 + 0.44108215673156714j)],
                    [(0.1639041643132761 - 0.6786784418985348j), (-0.04553731046489114 - 0.07287127470610173j)],
                    [(0.1639041643132761 + 0.6786784418985348j), (-0.17806561332820908 + 0.2918483705626999j)],
                ],
                [
                    [(-0.30133737241459735 + 0j), (-0.28272877510796823 - 0.2546148613447606j)],
                    [(0.1994203879613906 - 0.08578241586968506j), (-0.43291412283763553 + 0.0414533485984747j)],
                    [(0.1994203879613906 + 0.08578241586968506j), (0.46880967842846877 - 0.09812887702412225j)],
                ],
            ],
            [
                [
                    [(-0.01681250924454277 + 0j), (-0.08659059967728863 + 0.17147654688529146j)],
                    [(-0.07274548965033673 + 0.5984893788062006j), (-0.36674605692950835 - 0.08744443768754959j)],
                    [(-0.07274548965033673 - 0.5984893788062006j), (-0.35403505983625405 - 0.22659276215149518j)],
                ],
                [
                    [(-0.3220317741152885 + 0j), (0.5656322935913487 - 0.211069809414898j)],
                    [(-0.1610196633716784 + 0.1268196478862158j), (-0.2200267066244847 + 0.04256581425069098j)],
                    [(-0.1610196633716784 - 0.1268196478862158j), (0.08825872431323079 + 0.019035668270447663j)],
                ],
                [
                    [(-0.11127007749668236 + 0j), (-0.14413660485460414 + 0.1443704619720686j)],
                    [(0.2196598314415092 - 0.16413695932382963j), (0.47539474319864217 + 0.39994060621510075j)],
                    [(0.2196598314415092 + 0.16413695932382963j), (-0.03212968582611006 + 0.06995611123794893j)],
                ],
            ],
            [
                [
                    [(-0.5572161138016851 + 0j), (-0.06784754068582544 + 0.06164564070537759j)],
                    [(-0.007301540934447098 - 0.2944502646390467j), (-0.36020839736540977 - 0.32951319722343153j)],
                    [(-0.007301540934447098 + 0.2944502646390467j), (-0.16704581358255455 + 0.004700073169605079j)],
                ],
                [
                    [(0.07097759978429702 + 0j), (0.00798023326909951 - 0.4615561726932731j)],
                    [(-0.4470742944340481 + 0.008264169732178956j), (0.3045623530710744 + 0.32317275977698046j)],
                    [(-0.4470742944340481 - 0.008264169732178956j), (-0.10718762044014732 - 0.3823784131214276j)],
                ],
                [
                    [(0.5866932384579262 + 0j), (0.054592712072217164 - 0.06988140098153256j)],
                    [(0.2414249829737221 + 0.03918267452385687j), (0.1793066515270914 + 0.4739644822299046j)],
                    [(0.2414249829737221 - 0.03918267452385687j), (0.03392166959529651 + 0.22267538534701856j)],
                ],
            ],
            [
                [
                    [(0.22974910940490328 + 0j), (-0.16079254068675686 - 0.06686417405295614j)],
                    [(-0.09850384960014628 - 0.26948640187923295j), (-0.07326506922234732 - 0.01910758692854178j)],
                    [(-0.09850384960014628 + 0.26948640187923295j), (-0.4344747892939895 - 0.4287311868948427j)],
                ],
                [
                    [(0.29278703398005335 + 0j), (0.4700290491444734 - 0.1343586318021185j)],
                    [(-0.22655332053280516 + 0.2263020063776794j), (0.3902473643942963 - 0.2677297935821991j)],
                    [(-0.22655332053280516 - 0.2263020063776794j), (0.16561181263505143 + 0.32713144601155447j)],
                ],
                [
                    [(0.5585170686670962 + 0j), (0.2288801464030653 - 0.2896956081477345j)],
                    [(-0.008743150007435335 - 0.068145165337759j), (0.059648728278050955 - 0.28418922443505046j)],
                    [(-0.008743150007435335 + 0.068145165337759j), (0.14229881350906665 + 0.2532084823952523j)],
                ],
            ],
        ]
    )
    obj.run(res=res, x=x_data)


@pytest.mark.api_fft_ihfft2_parameters
def test_ihfft2_2():
    """
    default
    x: tensor-3d
    s = (1, 2)
    """
    np.random.seed(33)
    x_data = np.random.rand(4, 3, 3) * 4 - 2
    res = np.array(
        [
            [[(-0.6030289030229492 + 0j), (-0.40293058722613107 + 0j)]],
            [[(0.9032298928329638 + 0j), (-0.9568773866308524 + 0j)]],
            [[(-0.2954985917455166 + 0j), (-0.10066444378056949 + 0j)]],
            [[(-0.6894504938047892 + 0j), (-1.087596427626529 + 0j)]],
        ]
    )
    obj.run(res=res, x=x_data, s=(1, 2))


@pytest.mark.api_fft_ihfft2_parameters
def test_ihfft2_3():
    """
    default
    x: tensor-3d
    s = (1, 2)
    axes=(0, 2)
    """
    np.random.seed(33)
    x_data = np.random.rand(4, 3, 3) * 4 - 2
    res = np.array(
        [
            [
                [(-0.6030289030229492 + 0j), (-0.40293058722613107 + 0j)],
                [(0.26139075846769866 + 0j), (-1.2201919949200992 + 0j)],
                [(-0.054173086132487835 + 0j), (-1.8671812121473361 + 0j)],
            ]
        ]
    )
    obj.run(res=res, x=x_data, s=(1, 2), axes=(0, 2))


@pytest.mark.api_fft_ihfft2_parameters
def test_ihfft2_4():
    """
    default
    x: tensor-3d
    s = (1, 2)
    axes=(0, 2)
    norm = 'forward'
    """
    np.random.seed(33)
    x_data = np.random.rand(4, 3, 3) * 4 - 2
    res = np.array(
        [
            [
                [(-1.2060578060458984 + 0j), (-0.8058611744522621 + 0j)],
                [(0.5227815169353973 + 0j), (-2.4403839898401984 + 0j)],
                [(-0.10834617226497567 + 0j), (-3.7343624242946722 + 0j)],
            ]
        ]
    )
    obj.run(res=res, x=x_data, s=(1, 2), axes=(0, 2), norm="forward")


@pytest.mark.api_fft_ihfft2_parameters
def test_ihfft2_5():
    """
    default
    x: tensor-3d
    s = (1, 2)
    axes=(0, 2)
    norm = 'ortho'
    """
    np.random.seed(33)
    x_data = np.random.rand(4, 3, 3) * 4 - 2
    res = np.array(
        [
            [
                [(-0.8528116531580245 + 0j), (-0.5698299011501499 + 0j)],
                [(0.36966235570400935 + 0j), (-1.7256120679150868 + 0j)],
                [(-0.07661231312417013 + 0j), (-2.640592993626998 + 0j)],
            ]
        ]
    )
    obj.run(res=res, x=x_data, s=(1, 2), axes=(0, 2), norm="ortho")