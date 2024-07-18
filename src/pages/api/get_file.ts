// Next.js API route support: https://nextjs.org/docs/api-routes/introduction
import type { NextApiRequest, NextApiResponse } from 'next'
import formidable from "formidable";
import { promises as fs } from 'fs';
import * as utils from '@/utils';

// Define the Response type
type Response = {
    code: number;
    file?: any;
    message?: string;
};

export const config = {
    basePath: '/sam',
    api: {
        bodyParser: false
    }
};

export default async function handler(
    req: NextApiRequest,
    res: NextApiResponse<Response>
) {
    fs.mkdir('./tmp/', { recursive: true })
    const form = formidable({ uploadDir: './tmp/', /*maxTotalFileSize: 2048 * 2048 */})
    const { fields } =
        await new Promise<{ fields: formidable.Fields}>((resolve, reject) => {
            form.parse(req, async function (err, fields, files) {
                if (err) {
                    reject(err);
                    return;
                }
                resolve({fields});
            });
        });
    const req_data = new FormData();
    req_data.append('UUID', fields['UUID'] as string);
    console.log(req_data);
    const res_data = await fetch(
        utils.config.API_URL + '/api/get_file',
        {
            method: 'POST',
            body: req_data,
        }
    );
    const res_data_json = await res_data.json();
    res.status(200).json(res_data_json);
}