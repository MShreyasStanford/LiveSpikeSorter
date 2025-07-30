#ifndef SORTPARAMETERS_H
#define SORTPARAMETERS_H
#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>

struct SorterParameters {
	long		m_lT = 0, // #Templates
				m_lC = 0, // #Channels
				m_lM = 0, // #Samples/Template
				m_lN = 0; // Size of batch = M + m_lMaxScanWin;

	float		m_fSampRate = 0; // Samplerate

	std::vector<double> m_dChanpos;
	std::vector<int>    m_vNeuronIndices;

	// Using the Cereal serialization library
	template <class Archive>
	void serialize(Archive & ar)
	{
		ar(m_lT,
			m_lC,
			m_lM,
			m_lN,
			m_fSampRate,
			m_dChanpos,
			m_vNeuronIndices);
	}
};
#endif
