#ifndef ONLINESPIKESPAYLOAD_H
#define ONLINESPIKESPAYLOAD_H
#include <cereal/archives/binary.hpp>
#include <cereal/types/array.hpp>
#include <cereal/types/vector.hpp>
#include "SerializationHelpers.h"

struct OnlineSpikesPayload {
	long        recordingOffset;
	long		streamSampleCt;
	
	std::vector<long> Times;
	std::vector<long> Templates;
	std::vector<float> Amplitudes;

	double		VRMS;
	float		P2P;
	long		processTime;	

	// These used for Decoder->GUI but not OnlineSpikes->Decoder 
	long eventStreamSampleCt;
	int16_t		predictLabel;
	int16_t		label;
	int16_t		nTrials;
	int16_t		nCorrect;
	double		confidence;


	// Using the Cereal serialization library
	template <class Archive>
	void serialize(Archive & ar)
	{
		ar(recordingOffset,
			streamSampleCt,
			Times,
			Templates,
			Amplitudes,
			VRMS,
			P2P,
			processTime,
			eventStreamSampleCt,
			predictLabel,
			label,
			nTrials,
			nCorrect,
			confidence);
	}
};
#endif
