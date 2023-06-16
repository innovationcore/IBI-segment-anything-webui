// Next.js API route support: https://nextjs.org/docs/api-routes/introduction
import type { NextApiRequest, NextApiResponse } from 'next'
import formidable from "formidable";
import { promises as fs } from 'fs';
import * as utils from '@/utils';

export const config = {
    basePath: '/sam',
    api: {
        bodyParser: false
    }
};


export default async function handler(
    req: NextApiRequest,
    res: NextApiResponse<Response>) {
    const res_data = await fetch(
        utils.config.API_URL + '/api/populate',
        {
            method: 'POST',
        }
    )
    const res_data_json = await res_data.json()
    res.status(200).json(res_data_json)
}