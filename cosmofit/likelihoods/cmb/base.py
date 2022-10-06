import numpy as np
from scipy import constants

from cosmofit.likelihoods.base import BaseCalculator


class ClickLikelihood(BaseCalculator):

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)
        self.need_cosmo_arguments(
            data, {'lensing': 'yes', 'output': 'tCl lCl pCl'})

        try:
            import clik
        except ImportError:
            raise io_mp.MissingLibraryError(
                "You must first activate the binaries from the Clik " +
                "distribution. Please run : \n " +
                "]$ source /path/to/clik/bin/clik_profile.sh \n " +
                "and try again.")
        # for lensing, some routines change. Intializing a flag for easier
        # testing of this condition
        #if self.name == 'Planck_lensing':
        if 'lensing' in self.name and 'Planck' in self.name:
            self.lensing = True
        else:
            self.lensing = False

        try:
            if self.lensing:
                self.clik = clik.clik_lensing(self.path_clik)
                try:
                    self.l_max = max(self.clik.get_lmax())
                # following 2 lines for compatibility with lensing likelihoods of 2013 and before
                # (then, clik.get_lmax() just returns an integer for lensing likelihoods;
                # this behavior was for clik versions < 10)
                except:
                    self.l_max = self.clik.get_lmax()
            else:
                self.clik = clik.clik(self.path_clik)
                self.l_max = max(self.clik.get_lmax())
        except clik.lkl.CError:
            raise io_mp.LikelihoodError(
                "The path to the .clik file for the likelihood "
                "%s was not found where indicated:\n%s\n"
                % (self.name,self.path_clik) +
                " Note that the default path to search for it is"
                " one directory above the path['clik'] field. You"
                " can change this behaviour in all the "
                "Planck_something.data, to reflect your local configuration, "
                "or alternatively, move your .clik files to this place.")
        except KeyError:
            raise io_mp.LikelihoodError(
                "In the %s.data file, the field 'clik' of the " % self.name +
                "path dictionary is expected to be defined. Please make sure"
                " it is the case in you configuration file")

        self.need_cosmo_arguments(
            data, {'l_max_scalars': self.l_max})

        self.nuisance = list(self.clik.extra_parameter_names)

        # line added to deal with a bug in planck likelihood release: A_planck called A_Planck in plik_lite
        if (self.name == 'Planck15_highl_lite') or (self.name == 'Planck15_highl_TTTEEE_lite'):
            for i in range(len(self.nuisance)):
                if (self.nuisance[i] == 'A_Planck'):
                    self.nuisance[i] = 'A_planck'
            print("In %s, MontePython corrected nuisance parameter name A_Planck to A_planck" % self.name)

        # testing if the nuisance parameters are defined. If there is at least
        # one non defined, raise an exception.
        exit_flag = False
        nuisance_parameter_names = data.get_mcmc_parameters(['nuisance'])
        for nuisance in self.nuisance:
            if nuisance not in nuisance_parameter_names:
                exit_flag = True
                print('%20s\tmust be a fixed or varying nuisance parameter' % nuisance)

        if exit_flag:
            raise io_mp.LikelihoodError(
                "The likelihood %s " % self.name +
                "expected some nuisance parameters that were not provided")

        # deal with nuisance parameters
        try:
            self.use_nuisance
        except:
            self.use_nuisance = []

        # Add in use_nuisance all the parameters that have non-flat prior
        for nuisance in self.nuisance:
            if hasattr(self, '%s_prior_center' % nuisance):
                self.use_nuisance.append(nuisance)

    def loglkl(self, cosmo, data):

        nuisance_parameter_names = data.get_mcmc_parameters(['nuisance'])

        # get Cl's from the cosmological code
        cl = self.get_cl(cosmo)

        # testing for lensing
        if self.lensing:
            try:
                length = len(self.clik.get_lmax())
                tot = np.zeros(
                    np.sum(self.clik.get_lmax()) + length +
                    len(self.clik.get_extra_parameter_names()))
            # following 3 lines for compatibility with lensing likelihoods of 2013 and before
            # (then, clik.get_lmax() just returns an integer for lensing likelihoods,
            # and the length is always 2 for cl['pp'], cl['tt'])
            except:
                length = 2
                tot = np.zeros(2*self.l_max+length + len(self.clik.get_extra_parameter_names()))
        else:
            length = len(self.clik.get_has_cl())
            tot = np.zeros(
                np.sum(self.clik.get_lmax()) + length +
                len(self.clik.get_extra_parameter_names()))

        # fill with Cl's
        index = 0
        if not self.lensing:
            for i in range(length):
                if (self.clik.get_lmax()[i] > -1):
                    for j in range(self.clik.get_lmax()[i]+1):
                        if (i == 0):
                            tot[index+j] = cl['tt'][j]
                        if (i == 1):
                            tot[index+j] = cl['ee'][j]
                        if (i == 2):
                            tot[index+j] = cl['bb'][j]
                        if (i == 3):
                            tot[index+j] = cl['te'][j]
                        if (i == 4):
                            tot[index+j] = 0 #cl['tb'][j] class does not compute tb
                        if (i == 5):
                            tot[index+j] = 0 #cl['eb'][j] class does not compute eb

                    index += self.clik.get_lmax()[i]+1

        else:
            try:
                for i in range(length):
                    if (self.clik.get_lmax()[i] > -1):
                        for j in range(self.clik.get_lmax()[i]+1):
                            if (i == 0):
                                tot[index+j] = cl['pp'][j]
                            if (i == 1):
                                tot[index+j] = cl['tt'][j]
                            if (i == 2):
                                tot[index+j] = cl['ee'][j]
                            if (i == 3):
                                tot[index+j] = cl['bb'][j]
                            if (i == 4):
                                tot[index+j] = cl['te'][j]
                            if (i == 5):
                                tot[index+j] = 0 #cl['tb'][j] class does not compute tb
                            if (i == 6):
                                tot[index+j] = 0 #cl['eb'][j] class does not compute eb

                        index += self.clik.get_lmax()[i]+1

            # following 8 lines for compatibility with lensing likelihoods of 2013 and before
            # (then, clik.get_lmax() just returns an integer for lensing likelihoods,
            # and the length is always 2 for cl['pp'], cl['tt'])
            except:
                for i in range(length):
                    for j in range(self.l_max):
                        if (i == 0):
                            tot[index+j] = cl['pp'][j]
                        if (i == 1):
                            tot[index+j] = cl['tt'][j]
                    index += self.l_max+1

        # fill with nuisance parameters
        for nuisance in self.clik.get_extra_parameter_names():

            # line added to deal with a bug in planck likelihood release: A_planck called A_Planck in plik_lite
            if (self.name == 'Planck15_highl_lite') or (self.name == 'Planck15_highl_TTTEEE_lite'):
                if nuisance == 'A_Planck':
                    nuisance = 'A_planck'

            if nuisance in nuisance_parameter_names:
                nuisance_value = data.mcmc_parameters[nuisance]['current'] *\
                    data.mcmc_parameters[nuisance]['scale']
            else:
                raise io_mp.LikelihoodError(
                    "the likelihood needs a parameter %s. " % nuisance +
                    "You must pass it through the input file " +
                    "(as a free nuisance parameter or a fixed parameter)")
            #print("found one nuisance with name",nuisance)
            tot[index] = nuisance_value
            index += 1

        # compute likelihood
        #print("lkl:",self.clik(tot))
        lkl = self.clik(tot)[0]

        # add prior on nuisance parameters
        lkl = self.add_nuisance_prior(lkl, data)

        # Option added by D.C. Hooper to deal with the joint prior on ksz_norm (A_ksz in Planck notation)
        # and A_sz (A_tsz in Planck notation), of the form ksz_norm + 1.6 * A_sz (according to eq. 23 of 1907.12875).
        # Behaviour (True/False), centre, and variance set in the .data files (default = True).

        # Check if the joint prior has been requested
        if getattr(self, 'joint_sz_prior', False):

            # Check that the joint_sz prior is only requested when A_sz and ksz_norm are present
            if not ('A_sz' in self.clik.get_extra_parameter_names() and 'ksz_norm' in self.clik.get_extra_parameter_names()):
                 raise io_mp.LikelihoodError(
                    "You requested a gaussian prior on ksz_norm + 1.6 * A_sz," +
                    "however A_sz or ksz_norm are not present in your param file.")

            # Recover the current values of the two sz nuisance parameters
            A_sz =  data.mcmc_parameters['A_sz']['current'] * data.mcmc_parameters['A_sz']['scale']
            ksz_norm = data.mcmc_parameters['ksz_norm']['current'] * data.mcmc_parameters['ksz_norm']['scale']

            # Combine the two into one new nuisance-like variable
            joint_sz = ksz_norm + 1.6 * A_sz

            # Check if the user has passed the prior center and variance on sz, otherwise abort
            if not (hasattr(self, 'joint_sz_prior_center') and hasattr(self, 'joint_sz_prior_variance')):
                raise io_mp.LikelihoodError(
                    " You requested a gaussian prior on ksz_norm + 1.6 * A_sz," +
                    " however you did not pass the center and variance." +
                    " You can pass this in the .data file.")

            # add prior on joint_sz parameter
            if not self.joint_sz_prior_variance == 0:
                lkl += -0.5*((joint_sz-self.joint_sz_prior_center)/self.joint_sz_prior_variance)**2

            # End of block for joint sz prior.

        return lkl
